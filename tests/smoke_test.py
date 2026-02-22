"""
Smoke test: runs the FULL pipeline end-to-end on a small model that fits on RTX 4060 (8GB).

Tests:
1. Build small model -> forward pass -> generate DSL tokens
2. DSL engine: parse tokens -> execute program -> validate
3. Refinement loop: generate -> execute -> validate -> refine (3 iterations)
4. Evaluation: score predictions against targets
5. Integration adapters: Causeway/BroadMind/FluxMind (if source available)

Usage:
    python tests/smoke_test.py
    python tests/smoke_test.py --device cpu   # force CPU
"""

import sys
import os
import time
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


def make_smoke_config():
    """Small config that fits on RTX 4060 with room to spare (~50M params)."""
    from config.model_config import ModelConfig
    return ModelConfig(
        hidden_dim=256,
        num_layers=8,
        attention_layer_positions=[3, 7],
        num_grid_colors=10,
        num_dsl_tokens=200,
        num_special_tokens=4,
        max_seq_len=512,
        num_query_heads=8,
        num_kv_heads=4,
        head_dim=32,
        mamba_d_state=32,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_dt_rank=32,
        num_experts=8,
        top_k=2,
        expert_dim=128,
        moe_aux_loss_weight=0.01,
        rms_norm_eps=1e-5,
        dropout=0.0,
    )


def section(name):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")


def passed(msg):
    print(f"  [PASS] {msg}")


def failed(msg, e=None):
    print(f"  [FAIL] {msg}")
    if e:
        print(f"         {e}")


def run_smoke_test(device='cuda'):
    print(f"ARC-AGI-2 Smoke Test")
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")

    results = {"passed": 0, "failed": 0, "skipped": 0}
    t_start = time.time()

    # ================================================================
    section("1. Model Build + Forward Pass")
    # ================================================================
    try:
        from config.model_config import ModelConfig
        from model.hybrid_arc import HybridARC, ModelOutput

        config = make_smoke_config()
        config.validate()
        model = HybridARC(config).to(device)

        counts = model.count_parameters()
        total_mb = counts['total'] * 4 / 1e6  # fp32 size
        print(f"  Model params: {counts['total']:,} ({total_mb:.0f} MB in fp32)")
        print(f"  Active/token: {counts['active_per_token']:,}")

        # Forward pass
        B, T = 2, 64
        token_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
        row_ids = torch.randint(0, 10, (B, T), device=device)
        col_ids = torch.randint(0, 10, (B, T), device=device)

        model.train()
        output = model(token_ids, row_ids, col_ids)

        assert isinstance(output, ModelOutput)
        assert output.logits.shape == (B, T, config.vocab_size)
        assert output.hidden_states.shape == (B, T, config.hidden_dim)
        assert output.aux_loss.dim() == 0
        passed(f"Forward pass: logits {tuple(output.logits.shape)}, aux_loss={output.aux_loss.item():.4f}")

        # Backward pass
        loss = output.logits.mean() + output.aux_loss
        loss.backward()
        passed(f"Backward pass: loss={loss.item():.4f}")

        # Generation
        model.eval()
        gen_ids = model.generate(token_ids[:1, :16], row_ids[:1, :16], col_ids[:1, :16],
                                  max_new_tokens=32, temperature=0.8, top_k=20)
        passed(f"Generation: {16} input tokens -> {gen_ids.shape[1]} total tokens")
        results["passed"] += 3

        if device == 'cuda':
            vram_used = torch.cuda.max_memory_allocated() / 1e9
            print(f"  Peak VRAM: {vram_used:.2f} GB")

    except Exception as e:
        failed("Model build/forward", e)
        traceback.print_exc()
        results["failed"] += 1
        return results  # can't continue without model

    # ================================================================
    section("2. DSL Engine: Primitives + Execution")
    # ================================================================
    try:
        from dsl.primitives import (rot90, hmirror, vmirror, recolor,
                                     find_objects, trim, PRIMITIVE_FUNCTIONS)
        from dsl.program import DSLProgram, DSLNode
        from dsl.engine import DSLEngine
        from dsl.parser import DSLParser
        from dsl.validator import DSLValidator
        from dsl.error_trace import build_error_traces_from_validation

        grid = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 2, 0],
        ])

        # Test primitives
        r = rot90(grid)
        assert r.shape == (5, 5)
        m = hmirror(grid)
        assert m.shape == (5, 5)
        assert not np.array_equal(grid, r)
        passed(f"{len(PRIMITIVE_FUNCTIONS)} primitives loaded, rot90/hmirror work")

        # Build a program: hmirror(rot90(input))
        prog = DSLProgram(root=DSLNode(op='hmirror', args=[
            DSLNode(op='rot90', args=[DSLNode(op='__input__', args=[])])
        ]))

        # Execute
        engine = DSLEngine()
        result = engine.execute(prog, grid)
        assert result.success
        assert result.output_grid is not None
        expected = hmirror(rot90(grid))
        assert np.array_equal(result.output_grid, expected)
        passed(f"Engine executed hmirror(rot90(input)): {result.output_grid.shape}")

        # Serialize -> parse roundtrip
        tokens = prog.to_tokens()
        parsed = DSLProgram.from_tokens(tokens)
        result2 = engine.execute(parsed, grid)
        assert result2.success
        assert np.array_equal(result.output_grid, result2.output_grid)
        passed(f"Token roundtrip: {len(tokens)} tokens, execution matches")

        # Validate against examples
        examples = [(grid, expected)]
        validator = DSLValidator()
        val_result = validator.validate(prog, examples)
        assert val_result.score == 1.0
        passed(f"Validator: score={val_result.score}")

        # Error trace on wrong program
        wrong_prog = DSLProgram(root=DSLNode(op='rot90', args=[
            DSLNode(op='__input__', args=[])
        ]))
        wrong_result = validator.validate(wrong_prog, examples)
        traces = build_error_traces_from_validation(wrong_result, examples)
        assert len(traces) > 0
        passed(f"Error trace: {traces[0].suggested_category}, {traces[0].summary[:50]}...")

        results["passed"] += 5

    except Exception as e:
        failed("DSL engine", e)
        traceback.print_exc()
        results["failed"] += 1

    # ================================================================
    section("3. Data Pipeline: Tokenizer + Synthetic Tasks")
    # ================================================================
    try:
        from data.grid_tokenizer import GridTokenizer
        from data.augmentation import augment_task
        from data.synthetic_tasks import SyntheticTaskGenerator

        tokenizer = GridTokenizer(max_grid_size=30, max_seq_len=512)

        # Encode a grid
        test_grid = np.array([[1, 2, 3], [4, 5, 6]])
        tids, rids, cids = tokenizer.encode_grid(test_grid)
        decoded = tokenizer.decode_grid(tids)
        assert np.array_equal(test_grid, decoded)
        passed(f"Grid tokenize roundtrip: {test_grid.shape} -> {len(tids)} tokens -> {decoded.shape}")

        # Encode a task
        demo_in = [grid]
        demo_out = [expected]
        encoded = tokenizer.encode_task(demo_in, demo_out, grid)
        assert 'token_ids' in encoded
        passed(f"Task encoding: {encoded['token_ids'].shape[0]} tokens")

        # Augmentation
        aug = augment_task(demo_in, demo_out)
        assert len(aug) >= 8  # at least 8 dihedral symmetries
        passed(f"Augmentation: {len(aug)} variants from 1 example")

        # Synthetic task generation
        gen = SyntheticTaskGenerator(num_demos=2, min_grid_size=3, max_grid_size=8,
                                     max_program_depth=2, seed=42)
        task = gen.generate_task()
        assert 'demo_inputs' in task
        assert 'program' in task
        assert len(task['demo_inputs']) == 2
        passed(f"Synthetic task: {len(task['demo_inputs'])} demos, "
               f"program='{task['program'].to_string()[:60]}'")

        results["passed"] += 4

    except Exception as e:
        failed("Data pipeline", e)
        traceback.print_exc()
        results["failed"] += 1

    # ================================================================
    section("4. Refinement Loop (3 iterations)")
    # ================================================================
    try:
        from refinement.budget_manager import BudgetManager, BudgetConfig
        from refinement.generator import DSLGenerator
        from refinement.loop import RefinementLoop

        # Budget for quick smoke test
        budget = BudgetConfig(
            max_iterations=3,
            max_time_seconds=30.0,
            confidence_threshold=0.99,
            pass_at_k=2,
            min_iterations=1,
            patience=3,
        )

        # Use the small model for generation
        generator = DSLGenerator(model=model, tokenizer_config=None, device=device)

        loop = RefinementLoop(
            model=model,
            generator=generator,
            ranker=None,
            fluxmind_validator=None,
            bridge=None,
            budget_config=budget,
            device=device,
        )

        # Create a simple task (identity-like)
        simple_grid = np.array([[1, 2], [3, 4]])
        simple_out = rot90(simple_grid)
        smoke_task = {
            'demo_inputs': [simple_grid],
            'demo_outputs': [simple_out],
            'test_input': simple_grid,
        }

        predictions = loop.solve(smoke_task)
        stats = loop.get_solve_stats()
        passed(f"Refinement loop ran: {stats.get('iterations_used', '?')} iterations, "
               f"{len(predictions)} predictions, "
               f"best_score={stats.get('best_score', 0):.2f}")
        results["passed"] += 1

    except Exception as e:
        failed("Refinement loop", e)
        traceback.print_exc()
        results["failed"] += 1

    # ================================================================
    section("5. Evaluation Metrics")
    # ================================================================
    try:
        from eval.metrics import (grid_exact_match, cell_accuracy,
                                   pass_at_k, structural_similarity,
                                   task_score, aggregate_scores)

        pred = np.array([[1, 2], [3, 4]])
        target = np.array([[1, 2], [3, 4]])
        wrong = np.array([[1, 2], [3, 5]])

        assert grid_exact_match(pred, target) == True
        assert grid_exact_match(pred, wrong) == False
        assert cell_accuracy(pred, target) == 1.0
        assert 0.0 < cell_accuracy(pred, wrong) < 1.0
        assert pass_at_k([pred], target, k=1) == True
        assert pass_at_k([wrong], target, k=1) == False
        assert pass_at_k([wrong, pred], target, k=2) == True

        ss = structural_similarity(pred, wrong)
        assert 0.0 <= ss <= 1.0

        ts = task_score([pred, wrong], target, k=2)
        assert ts['pass_at_k'] == True

        agg = aggregate_scores([ts])
        assert 'pass_at_k_rate' in agg
        passed(f"All metrics work: cell_acc={cell_accuracy(pred, wrong):.2f}, "
               f"structural_sim={ss:.2f}, pass@2={ts['pass_at_k']}")
        results["passed"] += 1

    except Exception as e:
        failed("Evaluation metrics", e)
        traceback.print_exc()
        results["failed"] += 1

    # ================================================================
    section("6. Training Loss")
    # ================================================================
    try:
        from training.losses import ARCLoss

        for stage in [1, 2, 3]:
            loss_fn = ARCLoss.for_stage(stage)
            model_out = {
                'logits': torch.randn(2, 32, config.vocab_size, device=device),
                'aux_loss': torch.tensor(0.1, device=device),
            }
            targets = {
                'target_tokens': torch.randint(0, config.vocab_size, (2, 32), device=device),
            }
            loss_dict = loss_fn(model_out, targets)
            assert 'total' in loss_dict
            assert not torch.isnan(loss_dict['total'])
        passed(f"Loss works for all 3 stages: stage3_total={loss_dict['total'].item():.4f}")
        results["passed"] += 1

    except Exception as e:
        failed("Training loss", e)
        traceback.print_exc()
        results["failed"] += 1

    # ================================================================
    section("7. Integration Adapters (optional)")
    # ================================================================
    try:
        from integration.causeway_adapter import CausewayAdapter
        adapter = CausewayAdapter(d_model=256, d_causal=32, d_action=32)
        adapter = adapter.to(device)
        h = torch.randn(2, 256, device=device)
        action = torch.randn(2, 32, device=device)
        delta = adapter(h, action)
        passed(f"CausewayAdapter: delta.overall_improvement shape={delta.overall_improvement.shape}")
        results["passed"] += 1
    except ImportError:
        print("  [SKIP] Causeway source not available")
        results["skipped"] += 1
    except Exception as e:
        failed("CausewayAdapter", e)
        results["failed"] += 1

    try:
        from integration.broadmind_adapter import BroadMindAdapter
        adapter = BroadMindAdapter(d_model=256, broadmind_d_model=192, num_dsl_ops=50)
        adapter = adapter.to(device)
        grid_emb = torch.randn(2, 256, device=device)
        ops = torch.randint(0, 50, (2, 4), device=device)
        result = adapter.execute_program(grid_emb, ops)
        passed(f"BroadMindAdapter: predictions shape={result.predictions.shape}")
        results["passed"] += 1
    except ImportError:
        print("  [SKIP] BroadMind source not available")
        results["skipped"] += 1
    except Exception as e:
        failed("BroadMindAdapter", e)
        results["failed"] += 1

    try:
        from integration.fluxmind_adapter import FluxMindAdapter
        adapter = FluxMindAdapter(d_model=256)
        grid_in = np.array([[1, 2], [3, 4]])
        grid_out = np.array([[4, 3], [2, 1]])
        score = adapter.score_program(['rot90'], [(grid_in, grid_out)])
        assert 0.0 <= score <= 1.0
        passed(f"FluxMindAdapter: score={score:.3f}")
        results["passed"] += 1
    except ImportError:
        print("  [SKIP] FluxMind source not available")
        results["skipped"] += 1
    except Exception as e:
        failed("FluxMindAdapter", e)
        results["failed"] += 1

    # ================================================================
    section("RESULTS")
    # ================================================================
    elapsed = time.time() - t_start
    total = results["passed"] + results["failed"]
    print(f"\n  Passed:  {results['passed']}/{total}")
    print(f"  Failed:  {results['failed']}/{total}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Time:    {elapsed:.1f}s")

    if device == 'cuda':
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak:.2f} GB")

    if results["failed"] == 0:
        print(f"\n  ALL TESTS PASSED")
    else:
        print(f"\n  SOME TESTS FAILED")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None,
                        help='Force device (cuda/cpu). Auto-detects if not set.')
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    results = run_smoke_test(device)
    sys.exit(1 if results["failed"] > 0 else 0)

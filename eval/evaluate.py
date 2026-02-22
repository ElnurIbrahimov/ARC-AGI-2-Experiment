"""
ArcEvaluator — full evaluation pipeline for ARC-AGI-2.

Runs the refinement loop on each task in the evaluation set,
collects predictions, and computes metrics.

CLI usage:
    python -m eval.evaluate \
        --checkpoint path/to/model.pt \
        --data_dir path/to/arc-agi-2 \
        --split evaluation \
        --max_iterations 500 \
        --max_time 300 \
        --pass_at_k 2 \
        --output_dir ./results \
        --visualize
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.model_config import ModelConfig
from data.arc_dataset import ARCDataset
from eval.metrics import (
    aggregate_scores,
    cell_accuracy,
    grid_exact_match,
    pass_at_k,
    structural_similarity,
    task_score,
)
from eval.visualize import (
    plot_refinement_progress,
    save_task_visualization,
)
from refinement.budget_manager import BudgetConfig
from refinement.generator import DSLGenerator
from refinement.loop import RefinementLoop
from refinement.ranker import CandidateRanker
from refinement.validator_fluxmind import FluxMindValidator

logger = logging.getLogger(__name__)


class ArcEvaluator:
    """
    Full evaluation pipeline for ARC-AGI-2.

    Runs the refinement loop on each task, collects predictions,
    computes metrics, and optionally saves visualizations.
    """

    def __init__(
        self,
        model,
        refinement_loop: Optional[RefinementLoop] = None,
        budget_config: Optional[BudgetConfig] = None,
        device: str = 'cuda',
        output_dir: Optional[str] = None,
        visualize: bool = False,
    ):
        """
        Args:
            model: HybridARC model instance (already loaded and on device).
            refinement_loop: Pre-built RefinementLoop. If None, one is
                             constructed from the model and budget_config.
            budget_config: BudgetConfig for the refinement loop. Ignored
                           if refinement_loop is provided.
            device: Torch device string.
            output_dir: Directory to save results and visualizations.
            visualize: Whether to generate per-task visualization PNGs.
        """
        self.model = model
        self.device = device
        self.output_dir = output_dir
        self.visualize = visualize
        self.budget_config = budget_config or BudgetConfig()

        if refinement_loop is not None:
            self.loop = refinement_loop
        else:
            self.loop = self._build_refinement_loop(model)

        # Create output directory if needed
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            if self.visualize:
                os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────

    def evaluate(self, dataset, max_tasks: Optional[int] = None) -> Dict:
        """
        Evaluate on a full dataset.

        Args:
            dataset: ARCDataset instance or list of task dicts.
                     Each task dict must have:
                       'demo_inputs', 'demo_outputs', 'test_input',
                       'test_output' (optional), 'task_id' (optional).
            max_tasks: Limit number of tasks (for quick eval runs).

        Returns:
            Dict with 'aggregate' metrics and 'per_task' results list.
        """
        # Normalize dataset to a list of task dicts
        tasks = self._normalize_dataset(dataset)
        if max_tasks is not None and max_tasks > 0:
            tasks = tasks[:max_tasks]

        num_tasks = len(tasks)
        logger.info("Starting evaluation on %d tasks", num_tasks)

        eval_start = time.time()
        per_task_results: List[Dict] = []
        task_scores_list: List[Dict] = []

        for idx, task in enumerate(tasks):
            task_id = task.get('task_id', f'task_{idx:04d}')
            logger.info("Evaluating task %d/%d: %s", idx + 1, num_tasks, task_id)

            try:
                result = self.evaluate_single_task(task)
                result['task_id'] = task_id
                per_task_results.append(result)

                # Build task_score dict for aggregate computation
                if result.get('target') is not None and result['predictions']:
                    ts = task_score(
                        result['predictions'],
                        result['target'],
                        k=self.budget_config.pass_at_k,
                    )
                    task_scores_list.append(ts)
                else:
                    # No target available — record placeholder
                    task_scores_list.append({
                        'pass_at_k': False,
                        'best_cell_accuracy': result.get('cell_accuracy', 0.0),
                        'best_shape_match': False,
                        'best_color_accuracy': 0.0,
                        'best_structural_similarity': result.get('structural_similarity', 0.0),
                    })

                # Log progress
                status = "PASS" if result.get('pass_at_2', False) else "FAIL"
                logger.info(
                    "  [%s] cell_acc=%.4f struct_sim=%.4f iters=%d time=%.1fs",
                    status,
                    result.get('cell_accuracy', 0.0),
                    result.get('structural_similarity', 0.0),
                    result.get('iterations_used', 0),
                    result.get('time_seconds', 0.0),
                )

                # Visualization
                if self.visualize and self.output_dir:
                    self._save_task_viz(task, result, task_id)

            except Exception as e:
                logger.error("Task %s failed with error: %s", task_id, e, exc_info=True)
                per_task_results.append({
                    'task_id': task_id,
                    'predictions': [],
                    'target': None,
                    'pass_at_2': False,
                    'exact_match': False,
                    'cell_accuracy': 0.0,
                    'structural_similarity': 0.0,
                    'iterations_used': 0,
                    'time_seconds': 0.0,
                    'error': str(e),
                })
                task_scores_list.append({
                    'pass_at_k': False,
                    'best_cell_accuracy': 0.0,
                    'best_shape_match': False,
                    'best_color_accuracy': 0.0,
                    'best_structural_similarity': 0.0,
                })

        total_time = time.time() - eval_start
        aggregate = aggregate_scores(task_scores_list)
        aggregate['total_time_seconds'] = round(total_time, 2)
        aggregate['avg_time_per_task'] = round(total_time / max(num_tasks, 1), 2)

        results = {
            'aggregate': aggregate,
            'per_task': per_task_results,
            'config': {
                'max_iterations': self.budget_config.max_iterations,
                'max_time_seconds': self.budget_config.max_time_seconds,
                'pass_at_k': self.budget_config.pass_at_k,
                'patience': self.budget_config.patience,
                'confidence_threshold': self.budget_config.confidence_threshold,
            },
        }

        self.print_summary(results)

        # Auto-save if output_dir is set
        if self.output_dir:
            self.save_results(results, os.path.join(self.output_dir, 'results.json'))

        return results

    def evaluate_single_task(self, task: Dict) -> Dict:
        """
        Evaluate on a single task.

        Args:
            task: dict with 'demo_inputs', 'demo_outputs', 'test_input',
                  and optionally 'test_output'.

        Returns:
            Dict with per-task metrics and predictions.
        """
        task_start = time.time()

        # Run the refinement loop
        predictions = self.loop.solve(task)
        solve_stats = self.loop.get_solve_stats()

        # Get the target output if available
        target = task.get('test_output')
        if target is not None:
            target = np.asarray(target, dtype=int)
            # Skip placeholder targets (1x1 zero grids from missing outputs)
            if target.shape == (1, 1) and target[0, 0] == 0:
                target = None

        # Compute metrics against target
        result: Dict = {
            'task_id': task.get('task_id', ''),
            'predictions': predictions,
            'target': target,
            'pass_at_2': False,
            'exact_match': False,
            'cell_accuracy': 0.0,
            'structural_similarity': 0.0,
            'iterations_used': solve_stats.get('iterations_used', 0),
            'time_seconds': round(time.time() - task_start, 2),
            'stopped_reason': solve_stats.get('stopped_reason', ''),
            'best_score': solve_stats.get('best_score', 0.0),
        }

        # Extract the best program's string representation
        if predictions:
            result['best_program'] = self._get_program_string(solve_stats)

        if target is not None and predictions:
            # Pass@2
            result['pass_at_2'] = pass_at_k(predictions, target, k=self.budget_config.pass_at_k)

            # Best prediction metrics
            best_cell_acc = 0.0
            best_struct_sim = 0.0
            best_exact = False

            for pred in predictions:
                ca = cell_accuracy(pred, target)
                ss = structural_similarity(pred, target)
                em = grid_exact_match(pred, target)

                if ca > best_cell_acc:
                    best_cell_acc = ca
                    best_struct_sim = ss
                    best_exact = em

            result['exact_match'] = best_exact
            result['cell_accuracy'] = best_cell_acc
            result['structural_similarity'] = best_struct_sim

        return result

    # ── Building ────────────────────────────────────────────────────────

    def _build_refinement_loop(self, model) -> RefinementLoop:
        """Build a refinement loop with available components."""
        generator = DSLGenerator(model=model, device=self.device)

        # Try to build optional components (they gracefully degrade if unavailable)
        ranker = CandidateRanker(model=model)
        fluxmind_validator = FluxMindValidator()

        loop = RefinementLoop(
            model=model,
            generator=generator,
            ranker=ranker,
            fluxmind_validator=fluxmind_validator,
            budget_config=self.budget_config,
            device=self.device,
        )

        return loop

    # ── Results I/O ─────────────────────────────────────────────────────

    def save_results(self, results: Dict, path: str) -> None:
        """Save evaluation results to JSON."""
        # Make a JSON-serializable copy
        serializable = _make_serializable(results)

        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(serializable, f, indent=2)

        logger.info("Results saved to %s", path)

    def print_summary(self, results: Dict) -> None:
        """Print evaluation summary to console."""
        agg = results.get('aggregate', {})
        config = results.get('config', {})

        num_tasks = agg.get('num_tasks', 0)
        pass_rate = agg.get('pass_at_k_rate', 0.0)
        num_passed = agg.get('num_passed', 0)
        mean_cell = agg.get('mean_cell_accuracy', 0.0)
        mean_struct = agg.get('mean_structural_similarity', 0.0)
        shape_rate = agg.get('shape_match_rate', 0.0)
        total_time = agg.get('total_time_seconds', 0.0)
        avg_time = agg.get('avg_time_per_task', 0.0)

        separator = "=" * 60
        print(f"\n{separator}")
        print("  ARC-AGI-2 Evaluation Results")
        print(separator)
        print(f"  Tasks evaluated:         {num_tasks}")
        print(f"  Pass@{config.get('pass_at_k', 2)} rate:             {pass_rate:.4f} ({num_passed}/{num_tasks})")
        print(f"  Mean cell accuracy:      {mean_cell:.4f}")
        print(f"  Mean structural sim:     {mean_struct:.4f}")
        print(f"  Shape match rate:        {shape_rate:.4f}")
        print(f"  Total time:              {total_time:.1f}s")
        print(f"  Avg time per task:       {avg_time:.1f}s")
        print(f"  Max iterations/task:     {config.get('max_iterations', 'N/A')}")
        print(f"  Max time/task:           {config.get('max_time_seconds', 'N/A')}s")
        print(separator)
        print()

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_dataset(dataset) -> List[Dict]:
        """Convert dataset to a list of task dicts."""
        if isinstance(dataset, list):
            return dataset

        # ARCDataset or any Dataset-like object
        tasks: List[Dict] = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            task = {
                'demo_inputs': item.get('demo_inputs', []),
                'demo_outputs': item.get('demo_outputs', []),
                'test_input': item.get('test_input', np.zeros((1, 1), dtype=int)),
                'test_output': item.get('test_output'),
                'task_id': item.get('task_id', f'task_{idx:04d}'),
            }
            tasks.append(task)
        return tasks

    @staticmethod
    def _get_program_string(solve_stats: Dict) -> str:
        """Extract a human-readable program string from solve stats if available."""
        # The solve_stats don't directly contain the program; this is a placeholder
        # that could be enriched if the loop tracks the best program object.
        return solve_stats.get('best_program', 'N/A')

    def _save_task_viz(self, task: Dict, result: Dict, task_id: str) -> None:
        """Save visualization for a single task."""
        try:
            viz_dir = os.path.join(self.output_dir, 'visualizations')

            # Task visualization: demo pairs + predictions
            all_inputs = list(task['demo_inputs']) + [task['test_input']]
            all_outputs = list(task['demo_outputs'])
            target = result.get('target')
            if target is not None:
                all_outputs.append(target)
            else:
                # Use a blank grid as placeholder
                all_outputs.append(np.zeros_like(task['test_input']))

            predictions = result.get('predictions', [])

            save_task_visualization(
                inputs=all_inputs,
                outputs=all_outputs,
                predictions=predictions[:1] if predictions else None,
                path=os.path.join(viz_dir, f'{task_id}.png'),
            )

            # Refinement progress plot
            iteration_scores = self.loop.get_solve_stats().get('iteration_scores', [])
            if iteration_scores and len(iteration_scores) > 1:
                fig = plot_refinement_progress(
                    iteration_scores,
                    title=f'Refinement Progress: {task_id}',
                )
                fig.savefig(
                    os.path.join(viz_dir, f'{task_id}_progress.png'),
                    dpi=100, bbox_inches='tight', facecolor='white',
                )
                import matplotlib.pyplot as plt
                plt.close(fig)

        except Exception as e:
            logger.debug("Visualization failed for %s: %s", task_id, e)


# ── Serialization helper ────────────────────────────────────────────────

def _make_serializable(obj):
    """Recursively convert numpy/torch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    return obj


# ── CLI Entry Point ─────────────────────────────────────────────────────

def main():
    """CLI entry point for ARC-AGI-2 evaluation."""
    parser = argparse.ArgumentParser(
        description='ARC-AGI-2 Evaluation Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to model checkpoint (.pt file)',
    )
    parser.add_argument(
        '--data_dir', type=str, required=True,
        help='Path to ARC-AGI-2 data root (contains training/, evaluation/ dirs)',
    )
    parser.add_argument(
        '--split', type=str, default='evaluation',
        choices=['training', 'evaluation'],
        help='Dataset split to evaluate on',
    )
    parser.add_argument(
        '--max_iterations', type=int, default=500,
        help='Max refinement iterations per task',
    )
    parser.add_argument(
        '--max_time', type=float, default=300.0,
        help='Max time (seconds) per task',
    )
    parser.add_argument(
        '--pass_at_k', type=int, default=2,
        help='Number of allowed submissions per task (ARC-AGI-2 = 2)',
    )
    parser.add_argument(
        '--patience', type=int, default=50,
        help='Stop if no improvement for this many iterations',
    )
    parser.add_argument(
        '--confidence_threshold', type=float, default=0.99,
        help='Stop early if confidence exceeds this value',
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Directory to save results and visualizations',
    )
    parser.add_argument(
        '--visualize', action='store_true',
        help='Generate per-task visualization PNGs',
    )
    parser.add_argument(
        '--max_tasks', type=int, default=None,
        help='Limit number of tasks (for quick eval)',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Torch device (cuda or cpu)',
    )
    parser.add_argument(
        '--log_level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging verbosity',
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    logger.info("ARC-AGI-2 Evaluation starting")
    logger.info("  Checkpoint:      %s", args.checkpoint)
    logger.info("  Data dir:        %s", args.data_dir)
    logger.info("  Split:           %s", args.split)
    logger.info("  Max iterations:  %d", args.max_iterations)
    logger.info("  Max time/task:   %.0fs", args.max_time)
    logger.info("  Pass@k:          %d", args.pass_at_k)
    logger.info("  Device:          %s", args.device)

    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Load model
    logger.info("Loading model from %s", args.checkpoint)
    model = _load_model(args.checkpoint, args.device)

    # Load dataset
    logger.info("Loading dataset from %s/%s", args.data_dir, args.split)
    dataset = ARCDataset(
        data_dir=args.data_dir,
        split=args.split,
        augment=False,
    )
    logger.info("Dataset contains %d task items", len(dataset))

    # Build budget config
    budget_config = BudgetConfig(
        max_iterations=args.max_iterations,
        max_time_seconds=args.max_time,
        pass_at_k=args.pass_at_k,
        patience=args.patience,
        confidence_threshold=args.confidence_threshold,
    )

    # Build evaluator
    evaluator = ArcEvaluator(
        model=model,
        budget_config=budget_config,
        device=args.device,
        output_dir=args.output_dir,
        visualize=args.visualize,
    )

    # Run evaluation
    results = evaluator.evaluate(
        dataset=dataset,
        max_tasks=args.max_tasks,
    )

    # Final summary
    agg = results['aggregate']
    logger.info(
        "Evaluation complete: Pass@%d = %.4f (%d/%d), mean_cell_acc=%.4f",
        args.pass_at_k,
        agg.get('pass_at_k_rate', 0.0),
        agg.get('num_passed', 0),
        agg.get('num_tasks', 0),
        agg.get('mean_cell_accuracy', 0.0),
    )

    return results


def _load_model(checkpoint_path: str, device: str):
    """
    Load a HybridARC model from a checkpoint.

    Handles both full model saves and state_dict saves.
    """
    from model.hybrid_arc import HybridARC

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine if this is a full save or state_dict save
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # State dict save with config
        config_dict = checkpoint.get('model_config', {})
        config = ModelConfig(**config_dict) if config_dict else ModelConfig()
        model = HybridARC(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Loaded model state dict from checkpoint")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Alternative key name
        config_dict = checkpoint.get('config', {})
        config = ModelConfig(**config_dict) if config_dict else ModelConfig()
        model = HybridARC(config)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("Loaded model state dict from checkpoint (alt key)")
    elif isinstance(checkpoint, HybridARC):
        # Full model save
        model = checkpoint
        logger.info("Loaded full model from checkpoint")
    else:
        # Assume it's a raw state dict
        config = ModelConfig()
        model = HybridARC(config)
        model.load_state_dict(checkpoint)
        logger.info("Loaded raw state dict from checkpoint")

    model = model.to(device)
    model.eval()

    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.2fM parameters", total_params / 1e6)

    return model


if __name__ == '__main__':
    main()

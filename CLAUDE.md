# ARC-AGI-2 Experiment

## Architecture
7B hybrid Mamba-2 + Transformer MoE model for ARC-AGI-2 benchmark.
- 32 layers: 24 Mamba-2 + 4 attention (at layers 7, 15, 23, 31)
- 64 experts, top-8 routing per token (~1B active params)
- Neurosymbolic refinement loop: generate DSL -> execute -> validate -> rank -> refine

## Key Directories
- `model/` — Neural network architecture
- `dsl/` — DSL primitives and symbolic executor
- `data/` — Dataset loading and preprocessing
- `refinement/` — Generate-execute-validate-refine loop
- `integration/` — Causeway, BroadMind, FluxMind adapters
- `training/` — Training scripts for 3 stages
- `eval/` — Evaluation and visualization

## Integration Modules
- Causeway (causal reasoning): `C:\Users\asus\Desktop\causeway`
- BroadMind (program execution): `C:\Users\asus\Desktop\BroadMind`
- FluxMind (meta-learning): `C:\Users\asus\Desktop\FluxMind paper MetaLearning\files`

## Training Stages
1. Pretrain on synthetic tasks (100K steps, 4xA100)
2. Finetune on ARC training set (20K steps)
3. Train integration modules only (10K steps, 7B frozen)

## Testing
Run tests: `python -m pytest tests/ -v`
Run single module: `python -m pytest tests/test_dsl.py -v`

## Key Interface Contracts
- Model: `HybridARC.forward(token_ids, row_ids, col_ids) -> ModelOutput(logits, hidden_states, aux_loss)`
- DSL: `DSLEngine.execute(program, grid) -> ExecutionResult(output_grid, success, error, trace)`
- Refinement: `RefinementLoop.solve(task) -> List[DSLProgram]`

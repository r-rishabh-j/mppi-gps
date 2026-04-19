# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the library code: `src/mppi/` for the MPPI planner, `src/gps/` for GPS and DAgger training loops, `src/policy/` for neural policies, `src/envs/` for task environments, and `src/utils/` for configs, evaluation, math, and device helpers. Use `scripts/` for runnable entry points such as training, evaluation, tuning, and plotting. Store tuned JSON presets in `configs/` and MuJoCo XML assets in `assets/`. Treat `experiments/`, `results/`, `checkpoints/`, and `data/` as generated-artifact directories unless a change explicitly updates shared outputs.

## Build, Test, and Development Commands
Install dependencies with `uv sync` (Python 3.11+). Common workflows:

- `python -m scripts.run_acrobot` or `python -m scripts.run_hopper` runs MPPI control on a single environment.
- `python -m scripts.run_gps --env acrobot --device auto` launches GPS distillation.
- `python -m scripts.run_dagger --env acrobot --device auto` runs DAgger training.
- `python -m scripts.eval_checkpoint --ckpt <path> --n-eval 10 --render` evaluates a saved policy.
- `python -m scripts.visualisation.plot_results --env acrobot` plots experiment outputs.

`commands.md` is the canonical command reference; update it whenever CLI flags, scripts, or run layouts change.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and modules, `PascalCase` for classes, and explicit type hints on public functions where practical. Keep numerical code vectorized with NumPy/Torch instead of adding per-step Python loops. Prefer small, task-focused dataclasses in `src/utils/config.py` for new runtime settings. No formatter or linter is configured in `pyproject.toml`, so match surrounding style closely and keep imports grouped and readable.

## Testing Guidelines
This repo currently uses script-level validation rather than a dedicated `tests/` package. For policy-supervision checks, run `python -m scripts.test_sl`; for quick regression coverage, use short smoke runs such as `python -m scripts.run_gps --env acrobot --gps-iters 3 --device cpu` or the documented DAgger smoke command in `commands.md`. Name new validation scripts `scripts/test_*.py` and keep them runnable from the repo root.

## Commit & Pull Request Guidelines
Recent commit messages are short, imperative, and lowercase (`fixes`, `hopper`, `deterministic policy`). Keep commits narrowly scoped and use the same style. Pull requests should state the environment or algorithm affected, list the commands used for validation, and include plots or videos only when behavior changes are user-visible.

# Agent Execution Rules

1. **Working Directory:** Always execute commands from the project root (`~/ADLProject2`). Never `cd` into subfolders for terminal executions.
2. **Environment Discipline:** All python commands MUST be prefixed with the virtual environment path: `./venv/bin/python`. Never use the global system python.
3. **Pip Integrity:** If a new library is needed, install it using `./venv/bin/pip` and immediately update the plan.md notes.
4. **Non-Interactive Execution:** Use `cmd /c` (if on Windows terminal) or ensure all Linux commands are non-interactive to prevent the "Running..." hang.
5. **Hardware Awareness:** Use `torch.device("cuda")` in all script logic to leverage the RTX 5070 Ti.
6. **No Data Loss:** Never delete the `results/`, `models/`, or `ray_results/` folders without explicit confirmation, as these contain hours of training progress.
7. **Progress Tracking:** Always check the plan.md file before starting a new task to ensure you are following the correct sequence.
8. **Error Handling:** If a command fails, check the error message and try to fix it. If you are unable to fix it, ask for help.
9. **Interpretability Required:** Every training loop MUST include a `SummaryWriter` for TensorBoard. Do not perform any training run without logging "Reward Decomposition" (Aggression vs. Preservation).
10. **Folder Discipline:** Follow the standardized directory structure:
    - `data/raw/` and `data/processed/` for datasets
    - `src/` for source code
    - `notebooks/` for exploratory analysis
    - `models/` for trained models and checkpoints
    - `docs/` for documentation
    - `results/` for output files (plots, metrics, logs)


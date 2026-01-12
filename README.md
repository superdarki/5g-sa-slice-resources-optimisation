# 5g-sa-slice-resources-optimisation

Deep Q-Network (DQN) training and evaluation for 5G standalone (SA) network slicing.
The agent learns how many guard channels (G) to reserve for URLLC traffic while serving
eMBB traffic, using a C-based stochastic simulator and a Gymnasium environment.

## What this project does
- Models a 5G slicing system with total resources S and guard channels G.
- Uses a C simulator (`simulation.c`) to estimate URLLC loss and eMBB queue metrics.
- Wraps the simulator in a Gymnasium environment (`environment.py`).
- Trains a DQN to select G based on traffic load and S (`dqn.py`, `train.py`).
- Exports learned policies into CSV matrices and heatmaps for analysis.

## Project layout
- `train.py`: main entry point (optuna, train, retrain, test, export).
- `dqn.py`: DQN network, replay buffer, training and evaluation utilities.
- `environment.py`: Gymnasium environment for slicing with variable S.
- `csim.py`: builds and loads the C simulator shared library via ctypes.
- `simulation.c`: stochastic simulator and CLI.
- `inference.py`: run a trained model for a single scenario.
- `train_export.sh`: helper loop for repeated retraining + export.
- `requirements.txt`: Python dependencies.

## Requirements
- Python 3.12+ required.
- A C compiler (`gcc` on Linux/macOS, MSVC or MinGW on Windows).
- Python packages in `requirements.txt`.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first time you import the environment (or run training), `csim.py` compiles
`simulation.c` into a shared library (`simulation.so`/`.dylib`/`.dll`) automatically.

## Training and evaluation
The `train.py` script supports several modes:
- `optuna`: hyperparameter search, saves best to `best_hparams.json`.
- `train`: train from scratch using saved hyperparameters.
- `retrain`: continue training from an existing model.
- `test`: evaluate a trained model on predefined scenarios.
- `full`: optuna then train (default).
- `export`: export G matrices and heatmaps for configured S values.

Examples:
```bash
# Hyperparameter search + training (default)
python train.py full

# Train with existing hparams
python train.py train --hparams-file best_hparams.json --model best_slicing_dqn.pth

# Continue training an existing model
python train.py retrain --model best_slicing_dqn.pth

# Test a trained model on predefined scenarios
python train.py test --model best_slicing_dqn.pth
```

### Export matrices and heatmaps
```bash
python train.py export --model best_slicing_dqn.pth --export-dir exports
```

This generates:
- `exports/G_matrix_S273.csv` and `exports/G_matrix_S273.png`
- `exports/Diff_matrix_S273.csv` and `exports/Diff_matrix_S273.png`
- Cached simulated best values: `exports/Simulated_G_best_S273.csv`  

You can change the S value(s) used by export in the `train.py` file `export_results`function.

The `train_export.sh` script runs repeated retraining and exports into
`exports/<run_id>/` directories (useful for iterative experiments).

## Inference
Run a trained model for a specific traffic scenario:
```bash
python inference.py \
  --model-path best_slicing_dqn.pth \
  --hparams-file best_hparams.json \
  --lambda-u 20 \
  --lambda-e 30 \
  --S 100 \
  --nb-iter 100000
```

## Simulator (optional standalone use)
You can run the C simulator directly:
```bash
gcc -shared -o simulation.so -fPIC simulation.c -lm -lpthread
./simulation 10.0 5.0 2.0 10.0 10 2 100000
```

## Outputs and artifacts
- `best_hparams.json`: best hyperparameters from optuna.
- `best_slicing_dqn.pth`: trained model weights (saved during training).
- `exports/`: CSV matrices and PNG heatmaps from export mode.

## Notes
- The environment state is `[urllc_load, embb_load, normalized_S]`.
- Actions map to a fraction of S, producing guard channels `G`.
- Rewards penalize URLLC SLA violations and favor minimal feasible G.

import argparse
import json
import os

import numpy as np
import torch
from gymnasium import spaces

from dqn import DQNAgent
from environment import SlicingEnv


def load_hparams(path):
    """Loads a dictionary of hyperparameters from a JSON file."""
    with open(path, "r") as f:
        hparams = json.load(f)
    print(f"Hyperparameters loaded from {path}")
    return hparams


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with a trained DQN model for 5G slicing."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model file (.pth).",
    )
    parser.add_argument(
        "--hparams-file",
        type=str,
        default="best_hparams.json",
        help="Path to save/load hyperparameters.",
    )
    parser.add_argument(
        "--lambda-u", type=float, required=True, help="Arrival rate for URLLC packets."
    )
    parser.add_argument(
        "--lambda-e", type=float, required=True, help="Arrival rate for eMBB packets."
    )
    parser.add_argument(
        "--S",
        type=int,
        default=50,
        help="Total number of resources (channels). Default: 50.",
    )
    parser.add_argument(
        "--nb-iter",
        type=int,
        default=1000000,
        help="Simulation iterations. Default: 1000000.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        hparams = load_hparams(args.hparams_file)
    except FileNotFoundError:
        print(f"Error: Hyperparameter file '{args.hparams_file}' not found.")
        return

    temp_agent = DQNAgent(state_dim=3, action_dim=21, hparams=hparams, device=device)
    try:
        # The load method now conveniently returns s_max for us
        loaded_s_max = temp_agent.load(args.model_path)
        print(
            f"Successfully loaded model from '{args.model_path}' (trained with S_max={loaded_s_max})"
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    env = SlicingEnv(S_max=loaded_s_max, NbIter=args.nb_iter)
    assert isinstance(env.action_space, spaces.Discrete)
    assert isinstance(env.observation_space, spaces.Box)

    agent = temp_agent
    agent.s_max = loaded_s_max

    # --- 4. Run Inference ---

    # Define the state based on input parameters
    urllc_load = args.lambda_u / env.mu_u
    embb_load = args.lambda_e / env.mu_e

    state = np.array([urllc_load, embb_load], dtype=np.float32)

    # Get the agent's decision (action)
    action = agent.select_action(state)

    # Set the traffic parameters in the environment for the simulation run
    env.lambda_u = args.lambda_u
    env.lambda_e = args.lambda_e

    # Run the simulation with the agent's chosen action
    _, _, _, _, info = env.step(action)
    chosen_g = info["G"]
    urllc_loss = info["urllc_loss"]
    embb_queue = info["embb_queue_avg"]

    # --- 4. Display Results ---
    print("\n" + "=" * 50)
    print("      5G Slicing RL Agent - Inference Results")
    print("=" * 50)
    print("\n[INPUT PARAMETERS]")
    print(f"  - URLLC Arrival Rate (lambda_u): {args.lambda_u:.2f}")
    print(f"  - eMBB Arrival Rate (lambda_e):  {args.lambda_e:.2f}")
    print(f"  - Total Channels (S):            {args.S}")
    print(
        f"  - Corresponding State (Loads):   [URLLC: {urllc_load:.2f}, eMBB: {embb_load:.2f}]"
    )

    print("\n[AGENT DECISION]")
    print(f"  - Chosen Guard Channels (G):     {chosen_g}")

    print("\n[SIMULATION OUTCOME]")
    print(f"  - URLLC Packet Loss:             {urllc_loss:.4e}")
    print(f"  - Target URLLC SLA:              <= {env.SLA_LOSS:.1e}")
    if urllc_loss <= env.SLA_LOSS:
        print("  - SLA Status:                  \033[92mPASSED\033[0m")  # Green text
    else:
        print("  - SLA Status:                  \033[91mFAILED\033[0m")  # Red text
    print(f"  - Average eMBB Queue Size:       {embb_queue:.4f}")
    print("=" * 50)

    # Clean up compiled C files
    env.close()


if __name__ == "__main__":
    main()

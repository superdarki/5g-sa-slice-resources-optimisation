import argparse
import json
import os
from typing import Any
import numpy as np
import pandas as pd  # For CSV export
import matplotlib.pyplot as plt  # For heatmap export

from gymnasium import spaces

from tqdm import tqdm

from dqn import DQNAgent, DQNTrainAgent, ReplayBuffer, evaluate_agent
from environment import SlicingEnv


def save_hparams(hparams: dict[str, Any], path: str):
    """Saves a dictionary of hyperparameters to a JSON file."""
    with open(path, "w") as f:
        json.dump(hparams, f, indent=4)
    print(f"Hyperparameters saved to {path}")


def load_hparams(path: str):
    """Loads a dictionary of hyperparameters from a JSON file."""
    with open(path, "r") as f:
        hparams = json.load(f)
    print(f"Hyperparameters loaded from {path}")
    return hparams


def train_agent(
    agent: DQNTrainAgent,
    env: SlicingEnv,
    model_save_path: str,
    total_episodes: int = 20000,
    eval_freq: int = 500,
):
    print("--- Evaluating starting model performance... ---")
    initial_score, initial_viol_rate, initial_g_distance = evaluate_agent(
        agent, env, num_episodes=200, num_runs_per_scenario=5
    )
    best_eval_score = initial_score
    print(
        f"** Initial score to beat: {best_eval_score:.2f} (Violation Rate: {initial_viol_rate:.2%}, G distance: {initial_g_distance:.2f})"
    )

    hparams = agent.hparams
    pbar = tqdm(range(total_episodes), desc="Training Progress")
    for episode in pbar:
        state, _ = env.reset()
        action = np.int64(agent.select_action(state))
        next_state, reward, done, _, _ = env.step(action)
        agent.store_transition(state, action, reward, done, next_state)
        if len(agent.buffer) > hparams["batch_size"]:
            agent.learn()
        if episode % hparams["target_update_freq"] == 0:
            agent.update_target_network()
        agent.update_epsilon(episode)

        if episode > 0 and episode % eval_freq == 0:
            current_score, violation_rate, avg_g_distance = evaluate_agent(
                agent, env, num_episodes=200, num_runs_per_scenario=5
            )
            if current_score > best_eval_score:
                pbar.set_description(
                    f"Eval: Viol Rate={violation_rate:.2%}, Avg G Dist={avg_g_distance:.2f}, Epsilon={agent.epsilon:.2f}"
                )
                pbar.write(
                    f"++ New best model saved at episode {episode} with score {current_score:.2f} (last: {best_eval_score:.2f})"
                )
                best_eval_score = current_score
                agent.save(model_save_path)
            else:
                pbar.write(
                    f"-- Skipping model at episode {episode} with score {current_score:.2f} (best: {best_eval_score:.2f})"
                )
                agent.load(model_save_path)

    current_score, violation_rate, avg_g_distance = evaluate_agent(
        agent, env, num_episodes=200, num_runs_per_scenario=5
    )
    if current_score > best_eval_score:
        pbar.write(
            f"++ New best model saved at episode {total_episodes} with score {current_score:.2f} (last: {best_eval_score:.2f})"
        )
        best_eval_score = current_score
        agent.save(model_save_path)
    else:
        pbar.write(
            f"-- Skipping model at episode {total_episodes} with score {current_score:.2f} (best: {best_eval_score:.2f})"
        )
        agent.load(model_save_path)

    print("--- Training Finished ---")
    print(f"Best model saved to {model_save_path}")
    return model_save_path


def load_or_simulate_best_g(
    env: SlicingEnv, S_values: list[int], output_dir: str
) -> dict[int, np.ndarray]:
    """
    For each S in S_values, load the precomputed matrix of G_best values from CSV.
    If it does not exist yet, run the simulation once to build it and persist it.
    """
    os.makedirs(output_dir, exist_ok=True)
    best_g_matrices: dict[int, np.ndarray] = {}

    for S in S_values:
        csv_path = os.path.join(output_dir, f"Simulated_G_best_S{S}.csv")
        if os.path.exists(csv_path):
            df_best = pd.read_csv(csv_path, index_col=0)  # type: ignore
            best_g_matrices[S] = df_best.to_numpy(dtype=int)
            print(f"Loaded cached simulated G_best matrix for S={S} from {csv_path}")
            continue

        best_matrix = np.zeros((S + 1, S + 1), dtype=int)
        total_iters = (S + 1) * (S + 1)
        with tqdm(
            total=total_iters, desc=f"Simulating G_best (S={S})", leave=True
        ) as pbar:
            for lambda_u in range(0, S + 1):
                for lambda_e in range(0, S + 1):
                    env.lambda_u = float(lambda_u)
                    env.lambda_e = float(lambda_e)
                    env.s = S

                    # Use the max-action to quickly reach a feasible point, then rely on env's search.
                    action = np.int64(env.num_action_fractions - 1)
                    _, _, _, _, info = env.step(action)
                    best_matrix[lambda_u, lambda_e] = info["G_best"]
                    pbar.update(1)

        df_best = pd.DataFrame(
            best_matrix, index=range(0, S + 1), columns=range(0, S + 1)
        )
        df_best.index.name = "lambda_u"
        df_best.columns.name = "lambda_e"
        df_best.to_csv(csv_path)
        print(f"Simulated and saved G_best matrix for S={S} to {csv_path}")
        best_g_matrices[S] = best_matrix

    return best_g_matrices


def export_results(agent: DQNAgent, env: SlicingEnv, output_dir: str):
    """
    Exports model outputs into crossed tables (matrices) of G values.
    Rows = lambda_u, Columns = lambda_e.
    Creates one CSV file and one heatmap PNG per S value (10, 50, 150, 200).
    """
    os.makedirs(output_dir, exist_ok=True)

    S_values = [273]

    simulated_best = load_or_simulate_best_g(env, S_values, output_dir)

    for S in S_values:
        G_matrix = np.zeros((S + 1, S + 1), dtype=int)
        Diff_matrix = np.zeros((S + 1, S + 1), dtype=int)
        best_matrix = simulated_best[S]

        total_iters = (S + 1) * (S + 1)
        with tqdm(total=total_iters, desc=f"S={S}", leave=True) as pbar:
            for lambda_u in range(0, S + 1):
                for lambda_e in range(0, S + 1):
                    env.lambda_u = float(lambda_u)
                    env.lambda_e = float(lambda_e)
                    env.s = S

                    state = np.array(
                        [
                            env.lambda_u / env.mu_u,
                            env.lambda_e / env.mu_e,
                            S / env.s_max,
                        ],
                        dtype=np.float32,
                    )
                    action = agent.select_action(state)
                    action_fraction = action / (env.num_action_fractions - 1)
                    G_value = int(round(action_fraction * S))
                    G_matrix[lambda_u, lambda_e] = G_value
                    Diff_matrix[lambda_u, lambda_e] = (
                        G_value - best_matrix[lambda_u, lambda_e]
                    )
                    pbar.update(1)

        # --- Save G matrix ---
        df_g = pd.DataFrame(G_matrix, index=range(0, S + 1), columns=range(0, S + 1))
        df_g.index.name = "lambda_u"
        df_g.columns.name = "lambda_e"
        csv_path_g = os.path.join(output_dir, f"G_matrix_S{S}.csv")
        df_g.to_csv(csv_path_g)
        print(f"Exported G matrix for S={S} to {csv_path_g}")

        plt.figure(figsize=(8, 6))  # type: ignore
        plt.imshow(  # type: ignore
            G_matrix, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=S
        )
        plt.colorbar(label="G")  # type: ignore
        plt.title(f"G Matrix Heatmap (S={S})")  # type: ignore
        plt.xlabel("lambda_e")  # type: ignore
        plt.ylabel("lambda_u")  # type: ignore
        plt.xticks(np.linspace(0, S, min(S + 1, 11), dtype=int))  # type: ignore
        plt.yticks(np.linspace(0, S, min(S + 1, 11), dtype=int))  # type: ignore
        png_path_g = os.path.join(output_dir, f"G_matrix_S{S}.png")
        plt.savefig(png_path_g, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()
        print(f"Exported heatmap for G (S={S}) to {png_path_g}")

        # --- Save Diff matrix ---
        df_diff = pd.DataFrame(
            Diff_matrix, index=range(0, S + 1), columns=range(0, S + 1)
        )
        df_diff.index.name = "lambda_u"
        df_diff.columns.name = "lambda_e"
        csv_path_diff = os.path.join(output_dir, f"Diff_matrix_S{S}.csv")
        df_diff.to_csv(csv_path_diff)
        print(f"Exported Diff matrix for S={S} to {csv_path_diff}")

        plt.figure(figsize=(8, 6))  # type: ignore
        plt.imshow(  # type: ignore
            Diff_matrix, origin="lower", cmap="coolwarm", aspect="auto", vmin=-S, vmax=S
        )
        plt.colorbar(label="G - G_best")  # type: ignore
        plt.title(f"Difference Heatmap (S={S})")  # type: ignore
        plt.xlabel("lambda_e")  # type: ignore
        plt.ylabel("lambda_u")  # type: ignore
        plt.xticks(np.linspace(0, S, min(S + 1, 11), dtype=int))  # type: ignore
        plt.yticks(np.linspace(0, S, min(S + 1, 11), dtype=int))  # type: ignore
        png_path_diff = os.path.join(output_dir, f"Diff_matrix_S{S}.png")
        plt.savefig(png_path_diff, dpi=150, bbox_inches="tight")  # type: ignore
        plt.close()
        print(f"Exported heatmap for Diff (S={S}) to {png_path_diff}")


def main():
    parser = argparse.ArgumentParser(
        description="Train, optimize, test, or export results for a DQN in 5G Slicing.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=["optuna", "train", "retrain", "test", "full", "export"],
        help=(
            "Script operational mode:\n"
            "  optuna   - Run hyperparameter search and save the best params.\n"
            "  train    - Train a new model from scratch using saved params.\n"
            "  retrain  - Load an existing model and continue training it.\n"
            "  test     - Test a trained model on predefined test cases.\n"
            "  full     - Run optuna then train immediately (default).\n"
            "  export   - Export model outputs to CSV and PNG heatmaps for different S values."
        ),
    )
    parser.add_argument("--hparams-file", type=str, default="best_hparams.json")
    parser.add_argument("--model", type=str, default="best_slicing_dqn.pth")
    parser.add_argument("--optuna-trials", type=int, default=20)
    parser.add_argument("--train-episodes", type=int, default=40000)
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--s-min", type=int, default=50)
    parser.add_argument("--s-max", type=int, default=300)
    parser.add_argument("--export-dir", type=str, default="exports")

    args = parser.parse_args()

    hparams = None
    env = None

    if args.train_episodes % args.eval_freq != 0:
        raise ValueError("Train episodes should be a multiple of eval freq")

    # --- Export Mode ---
    if args.mode == "export":
        print("--- Exporting Model Outputs to CSV + Heatmaps ---")
        try:
            hparams = load_hparams(args.hparams_file)
        except FileNotFoundError:
            print(f"Error: Hyperparameter file '{args.hparams_file}' not found.")
            return

        env = SlicingEnv(S_min=args.s_min, S_max=args.s_max)
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.observation_space, spaces.Box)

        agent = DQNAgent(
            env.observation_space.shape[0], int(env.action_space.n), hparams
        )

        try:
            agent.load(args.model)
        except FileNotFoundError:
            print(f"Error: Model file '{args.model}' not found.")
            env.close()
            return

        export_results(agent, env, args.export_dir)
        env.close()

    # --- Mode: Optuna ---
    if args.mode in ["optuna", "full"]:
        import optuna

        def optuna_objective(trial: optuna.Trial) -> float:
            env = SlicingEnv(S_min=10, S_max=200)
            hparams: dict[str, Any] = {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-2, log=True
                ),
                "gamma": trial.suggest_float("gamma", 0.9, 0.999),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128, 256]
                ),
                "buffer_size": trial.suggest_categorical(
                    "buffer_size", [10000, 50000, 100000]
                ),
                "epsilon_decay": trial.suggest_int("epsilon_decay", 500, 2000),
                "target_update_freq": trial.suggest_int("target_update_freq", 10, 100),
                "hidden_layers": trial.suggest_int("hidden_layers", 2, 6),
                "hidden_size": trial.suggest_categorical(
                    "hidden_size", [64, 128, 256, 512]
                ),
            }
            hparams["epsilon_start"] = 1.0
            hparams["epsilon_end"] = 0.05
            buffer = ReplayBuffer(hparams["buffer_size"])
            assert isinstance(env.observation_space, spaces.Box)
            assert isinstance(env.action_space, spaces.Discrete)
            agent = DQNTrainAgent(
                env.observation_space.shape[0],
                int(env.action_space.n),
                buffer,
                hparams,
                env.s_max,
            )
            num_episodes = 5000
            pbar = tqdm(range(num_episodes), desc=f"Trial {trial.number}", leave=False)
            for episode in pbar:
                state, _ = env.reset()
                action = np.int64(agent.select_action(state))
                next_state, reward, done, _, _ = env.step(action)
                agent.store_transition(state, action, reward, done, next_state)
                if len(agent.buffer) > hparams["batch_size"]:
                    agent.learn()
                if episode % hparams["target_update_freq"] == 0:
                    agent.update_target_network()
                agent.update_epsilon(episode)
            pbar.close()
            score, _, _ = evaluate_agent(
                agent, env, num_episodes=200, num_runs_per_scenario=3
            )
            env.close()
            return score

        print(f"--- Running Optuna for {args.optuna_trials} Trials ---")
        study = optuna.create_study(direction="maximize", study_name="dqn_5g_slicing")
        study.optimize(optuna_objective, n_trials=args.optuna_trials, n_jobs=1)

        print("\n--- Optuna Optimization Finished ---")
        best_trial = study.best_trial
        print(f"Best trial score: {best_trial.value:.4f}")

        hparams = best_trial.params
        hparams["epsilon_start"] = 1.0
        hparams["epsilon_end"] = 0.05
        save_hparams(hparams, args.hparams_file)

    # --- Mode: Train, Retrain, or Full ---
    if args.mode in ["train", "retrain", "full"]:
        # Step 1: Get Hyperparameters
        try:
            hparams = load_hparams(args.hparams_file)
        except FileNotFoundError:
            print(f"Error: Hyperparameter file '{args.hparams_file}' not found.")
            return

        # Step 2: Setup Agent
        env = SlicingEnv(S_min=args.s_min, S_max=args.s_max)
        buffer = ReplayBuffer(hparams["buffer_size"])
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.observation_space, spaces.Box)
        agent = DQNTrainAgent(
            env.observation_space.shape[0],
            int(env.action_space.n),
            buffer,
            hparams,
            env.s_max,
        )

        # Step 3: Load existing weights if retraining
        if args.mode == "retrain":
            print(f"--- [Retrain Mode] ---")
            if not os.path.exists(args.model):
                print(f"Error: Model file '{args.model}' not found for retraining.")
                env.close()
                return

            ###   BACKUP   ###
            # i = 0
            # while os.path.exists(f"{args.model}.{i}.bak"):
            #     i += 1
            # backup_path = f"{args.model}.{i}.bak"
            # print(f"Backing up current model '{args.model}' to '{backup_path}'...")
            # try:
            #     shutil.copy(args.model, backup_path)
            #     print("Backup successful.")
            # except Exception as e:
            #     print(f"Error creating backup: {e}")
            #     env.close()
            #     return
            ### BACKUP END ###

            agent.load(args.model)
            print(
                f"Successfully loaded model from '{args.model}' to continue training."
            )
        else:
            print(f"--- [Train Mode] Starting training from scratch. ---")

        # Step 4: Run training loop
        train_agent(
            agent,
            env,
            args.model,
            total_episodes=args.train_episodes,
            eval_freq=args.eval_freq,
        )

    # --- Test the Model ---
    if args.mode in ["test", "train", "retrain", "full"]:
        print("\n--- Testing the Best Trained Model ---")

        # For a standalone 'test' run, we need to load hparams.
        if hparams is None:
            try:
                hparams = load_hparams(args.hparams_file)
            except FileNotFoundError:
                print(
                    f"Error: Hyperparameter file '{args.hparams_file}' not found for testing."
                )
                return

        # If the environment wasn't created during a training run, create it now.
        if env is None:
            env = SlicingEnv(S_min=args.s_min, S_max=args.s_max)
        assert isinstance(env.action_space, spaces.Discrete)
        assert isinstance(env.observation_space, spaces.Box)

        test_agent = DQNAgent(
            env.observation_space.shape[0], int(env.action_space.n), hparams
        )

        try:
            test_agent.load(args.model)
        except FileNotFoundError:
            print(f"Error: Model file '{args.model}' not found.")
            env.close()
            return

        test_scenarios: dict[str, dict[str, float]] = {
            "Low Load (S=20)": {"lambda_u": 2.0, "lambda_e": 5.0, "S": 20},
            "High Load (S=50)": {"lambda_u": 12.0, "lambda_e": 40.0, "S": 50},
            "URLLC Dom (S=100)": {"lambda_u": 40.0, "lambda_e": 10.0, "S": 100},
            "eMBB Dom (S=150)": {"lambda_u": 10.0, "lambda_e": 120.0, "S": 150},
            "Overload (S=50)": {"lambda_u": 120.0, "lambda_e": 20.0, "S": 50},
            "Test": {"lambda_u": 50.0, "lambda_e": 50.0, "S": 100},  # G=8
        }

        print(
            "\n{:<20} | {:<10} | {:<15} | {:<15}".format(
                "Scenario", "G / Best G", "URLLC Loss", "eMBB Queue Avg"
            )
        )
        print("-" * 70)

        for name, params in test_scenarios.items():
            current_S = int(params["S"])
            env.lambda_u = params["lambda_u"]
            env.lambda_e = params["lambda_e"]
            env.s = current_S

            state = np.array(
                [
                    env.lambda_u / env.mu_u,
                    env.lambda_e / env.mu_e,
                    current_S / env.s_max,
                ],
                dtype=np.float32,
            )
            action = np.int64(test_agent.select_action(state))
            _, _, _, _, info = env.step(action)
            print(
                "{:<20} | {:<10} | {} {:<11.2e} | {:<15.2f}".format(
                    name,
                    f"{info["G"]} / {info["G_best"]}",
                    f"({"V" if info["urllc_loss"]<1e-5 else "F"})",
                    info["urllc_loss"],
                    info["embb_queue_avg"],
                )
            )

    # Clean up the environment at the very end
    if env:
        env.close()


if __name__ == "__main__":
    main()

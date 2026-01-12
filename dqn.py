import collections
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

from environment import SlicingEnv


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer: collections.deque[
            tuple[np.ndarray, np.int64, float, bool, np.ndarray]
        ] = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience: tuple[np.ndarray, np.int64, float, bool, np.ndarray]):
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(next_states),
        )


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: int = 2,
        hidden_size: int = 128,
    ):
        """
        Initializes a Q-Network with a flexible architecture.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Dimension of the action space.
            hidden_layers (int): Number of hidden layers.
            hidden_size (int): Number of neurons in each hidden layer.
        """
        super().__init__()  # type: ignore

        layers: list[nn.Module] = []
        # Input Layer
        layers.append(nn.Linear(state_dim, hidden_size))
        layers.append(nn.ReLU())

        # Hidden Layers
        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output Layer
        layers.append(nn.Linear(hidden_size, action_dim))

        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hparams: dict[str, Any],
        device: str = "cpu",
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.policy_net = QNetwork(
            state_dim, action_dim, hparams["hidden_layers"], hparams["hidden_size"]
        ).to(device)
        self.s_max = None

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def load(self, path: str) -> Any:
        map_location = torch.device(
            "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
        )
        checkpoint = torch.load(path, map_location=map_location)

        self.policy_net.load_state_dict(checkpoint["model_state_dict"])
        self.policy_net.eval()

        loaded_s_max = checkpoint.get("s_max")
        if loaded_s_max is None:
            raise ValueError(
                "Model file is missing the 's_max' parameter. Please use a model trained with a newer script."
            )

        self.s_max = loaded_s_max
        return loaded_s_max


class DQNTrainAgent(DQNAgent):
    policy_net: nn.Module
    target_net: nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        replay_buffer: ReplayBuffer,
        hparams: dict[str, Any],
        s_max: int,
    ):
        super().__init__(
            state_dim,
            action_dim,
            hparams,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.buffer = replay_buffer
        self.hparams = hparams
        self.s_max = s_max

        self.target_net = QNetwork(
            state_dim, action_dim, hparams["hidden_layers"], hparams["hidden_size"]
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=hparams["learning_rate"]
        )

        self.epsilon = hparams["epsilon_start"]

    def select_action(self, state: np.ndarray) -> int:
        """
        Overrides the parent method to include epsilon-greedy exploration.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return super().select_action(state)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.int64,
        reward: float,
        done: bool,
        next_state: np.ndarray,
    ):
        self.buffer.append((state, action, reward, done, next_state))

    def update_epsilon(self, episode: int):
        eps_start = self.hparams["epsilon_start"]
        eps_end = self.hparams["epsilon_end"]
        eps_decay = self.hparams["epsilon_decay"]
        self.epsilon = eps_end + (eps_start - eps_end) * np.exp(
            -1.0 * episode / eps_decay
        )

    def learn(self):
        if len(self.buffer) < self.hparams["batch_size"]:
            return

        states, actions, rewards, dones, next_states = self.buffer.sample(
            self.hparams["batch_size"]
        )
        dones = torch.as_tensor(dones, dtype=torch.bool, device=self.device).unsqueeze(
            1
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        ).unsqueeze(1)
        actions = torch.as_tensor(
            actions, dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )

        current_q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

        not_done = (~dones).float()
        target_q_values = (
            rewards + not_done * float(self.hparams["gamma"]) * next_q_values
        )
        loss = F.smooth_l1_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path: str):
        save_dict: dict[str, Any] = {
            "model_state_dict": self.policy_net.state_dict(),
            "s_max": self.s_max,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> Any:
        loaded_s_max = super().load(path)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        return loaded_s_max


def calculate_performance_score(violation_rate: float, avg_g_distance: float) -> float:
    return -violation_rate * 10000 - avg_g_distance


def evaluate_agent(
    agent: DQNAgent,
    env: SlicingEnv,
    num_episodes: int = 100,
    num_runs_per_scenario: int = 5,
) -> tuple[float, float, float]:
    total_violations = 0
    total_g_distance = 0
    total_successful_scenarios = 0

    original_epsilon = 0.0
    if isinstance(agent, DQNTrainAgent):
        original_epsilon = getattr(agent, "epsilon", 0.0)
        agent.epsilon = 0.0

    pbar_eval = tqdm(range(num_episodes), desc="Evaluating Agent", leave=False)
    for _ in pbar_eval:
        state, _ = env.reset()
        action = np.int64(agent.select_action(state))

        run_losses: list[float] = []
        run_g_distances: list[float] = []

        for _ in range(num_runs_per_scenario):
            _, _, _, _, info = env.step(action)

            run_losses.append(info["urllc_loss"])
            if info["urllc_loss"] <= env.SLA_LOSS and info["G_best"] != -1:
                run_g_distances.append(abs(info["G"] - info["G_best"]))

        avg_loss = np.mean(run_losses)

        if avg_loss > env.SLA_LOSS:
            if env.lambda_u / env.mu_u <= env.s:
                total_violations += 1
        else:
            if len(run_g_distances) > 0:
                total_g_distance += np.mean(run_g_distances)
            total_successful_scenarios += 1

    if isinstance(agent, DQNTrainAgent):
        agent.epsilon = original_epsilon

    violation_rate = total_violations / num_episodes
    avg_g_distance = (
        float(total_g_distance / total_successful_scenarios)
        if total_successful_scenarios > 0
        else float("inf")
    )

    score = calculate_performance_score(violation_rate, avg_g_distance)
    return score, violation_rate, avg_g_distance

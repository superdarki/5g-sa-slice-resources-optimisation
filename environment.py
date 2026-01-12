import ctypes
import random
from typing import Any
import gymnasium as gym
import numpy as np
from csim import CSimResult, C_LIB


class SlicingEnv(gym.Env[np.ndarray, np.int64]):
    """
    Gym environment for the 5G slicing problem with a VARIABLE number of servers (S).

    - State: [urllc_load, embb_load, normalized_S]
    - Action: A discrete choice representing the FRACTION of S to use as guard channels.
    """

    def __init__(
        self,
        S_min: int = 10,
        S_max: int = 100,
        mu_e: float = 1.0,
        mu_u: float = 2.0,
        NbIter: int = 100000,
        sla_loss: float = 1e-5,
    ):
        super(SlicingEnv, self).__init__()

        # System parameters
        self.s_min = S_min
        self.s_max = S_max
        self.mu_e = mu_e
        self.mu_u = mu_u
        self.NbIter = NbIter
        self.SLA_LOSS = sla_loss

        # Episode-specific parameters (will be set in reset)
        self.s = 0
        self.lambda_e = 0
        self.lambda_u = 0

        # E.g., 21 actions for 0%, 5%, 10%, ..., 100%
        self.num_action_fractions = 21
        self.action_space = gym.spaces.Discrete(self.num_action_fractions)

        # State: [urllc_load, embb_load, normalized_S]
        # We normalize S by S_max for stable learning
        high = np.array([self.s_max * 1.5, self.s_max * 1.5, 1.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=high, shape=(3,), dtype=np.float32
        )

    def _generate_random_loads(self):
        """
        Generates random loads based on the CURRENT episode's S value.
        """
        self.lambda_u = random.uniform(0, self.s)
        self.lambda_e = random.uniform(0, self.s)

    def reset(
        self, *args: Any, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(*args, seed=seed, options=options)

        self.s = random.randint(self.s_min, self.s_max)

        self._generate_random_loads()

        urllc_load = self.lambda_u / self.mu_u
        embb_load = self.lambda_e / self.mu_e
        s_normalized = self.s / self.s_max

        state = np.array([urllc_load, embb_load, s_normalized], dtype=np.float32)

        info: dict[str, Any] = {
            "S": self.s,
            "lambda_u": self.lambda_u,
            "lambda_e": self.lambda_e,
        }
        return state, info

    def step(
        self, action: np.int64, simulate: bool = True
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        memo: dict[int, tuple[float, float]] = {}

        def run_sim_memoized(g_value: int) -> tuple[float, float]:
            if g_value in memo:
                return memo[g_value]
            res = CSimResult()
            C_LIB.simu(
                self.lambda_e,
                self.lambda_u,
                self.mu_e,
                self.mu_u,
                self.s,
                g_value,
                self.NbIter,
                ctypes.byref(res),
            )
            result = (res.loss, res.wait_avg)
            memo[g_value] = result
            return result

        action_fraction = action / (self.num_action_fractions - 1)
        G_chosen = int(round(action_fraction * self.s))

        if not simulate:
            next_state = np.array(
                [
                    self.lambda_u / self.mu_u,
                    self.lambda_e / self.mu_e,
                    self.s / self.s_max,
                ],
                dtype=np.float32,
            )
            info = {"G": G_chosen, "S": self.s}
            return next_state, 0.0, True, False, info

        loss_chosen, queue_chosen = run_sim_memoized(G_chosen)
        g_min_valid = self.s

        if loss_chosen <= self.SLA_LOSS:
            low, high = 0, G_chosen
            g_min_valid = G_chosen
            while low <= high:
                mid = low + (high - low) // 2
                loss_mid, _ = run_sim_memoized(mid)
                if loss_mid <= self.SLA_LOSS:
                    g_min_valid = mid
                    high = mid - 1
                else:
                    low = mid + 1

        if loss_chosen > self.SLA_LOSS:
            reward = -1 - np.log10(1 + (loss_chosen / self.SLA_LOSS))
        else:
            distance = abs(G_chosen - g_min_valid)
            normalized_distance = distance / self.s
            reward = 1.0 / (1.0 + normalized_distance * 10)

        terminated = True
        truncated = False
        s_normalized = self.s / self.s_max
        next_state = np.array(
            [self.lambda_u / self.mu_u, self.lambda_e / self.mu_e, s_normalized],
            dtype=np.float32,
        )

        info: dict[str, Any] = {
            "urllc_loss": loss_chosen,
            "embb_queue_avg": queue_chosen,
            "G": G_chosen,
            "S": self.s,
            "reward": reward,
            "G_best": g_min_valid,
        }
        return next_state, reward, terminated, truncated, info

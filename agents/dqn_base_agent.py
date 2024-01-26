from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import tensorflow as tf
from keras.losses import Huber
from keras.models import Model
from keras.optimizers import Adam
from tqdm import tqdm

from environments.base_environment import Timestep

if TYPE_CHECKING:
    from agents.configurations import EvalConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import ENV, RNG


class NNAgent:
    """soon"""

    models: list[tf.keras.models.Model]
    eval_model: tf.keras.models.Model

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: ENV,
        eval_config: EvalConfig,
        max_steps: int,
        seed: int,
    ) -> None:
        """soon"""
        self._rng: RNG = np.random.default_rng(seed)
        self.alpha: float = alpha
        self.epsilon: DecayParameter = epsilon
        self.gamma: float = gamma
        self._init_models(n_weights)
        self.optimizer: Adam = Adam(learning_rate=alpha)
        self.huber_loss: Huber = Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.n_weights: np.ndarray = np.arange(n_weights)
        self.env: ENV = env
        self.env_actions: np.ndarray = np.arange(4)
        self.eval: EvalConfig = eval_config
        self.seed: int = seed
        self.max_timesteps: int = max_steps
        self.mean_rewards: list[list[float]] = []
        self.rewards: list[list[float]] = []
        self.td_errors: list[list[float]] = []
        self.rolling_mean: deque[float] = deque(maxlen=100)

    def _init_models(self, n_weights: int) -> None:
        """soon"""
        input_layer = tf.keras.layers.Input(shape=(8))
        layer_1 = tf.keras.layers.Dense(512, activation="relu")(input_layer)
        layer_2 = tf.keras.layers.Dense(256, activation="relu")(layer_1)
        output_layer = tf.keras.layers.Dense(4, activation="linear")(layer_2)
        self.models = [
            Model(inputs=[input_layer], outputs=[output_layer])
            for _ in range(n_weights)
        ]
        self.eval_model = Model(inputs=[input_layer], outputs=[output_layer])
        self._get_eval_model()

    def reset_models(self) -> None:
        """soon"""
        self._init_models(len(self.n_weights))
        self.optimizer = Adam(learning_rate=self.alpha)

    def update_model(
        self,
        model: tf.keras.models.Model,
        states: np.ndarray,
        targets: np.ndarray,
        training: bool = True,
    ) -> float:
        """soon"""
        with tf.GradientTape() as tape:
            value_predictions = model(states, training=training)
            loss = self.huber_loss(value_predictions, targets)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return float(loss)

    def _get_eval_model(self) -> None:
        """soon"""
        if len(self.models) == 1:
            self.eval_model.set_weights(self.models[0].get_weights())
            return None
        weights_list = [model.get_weights() for model in self.models]
        weights = np.array(weights_list, dtype=object)
        self.eval_model.set_weights(np.mean(weights, axis=0))

    def _env_step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """soon"""
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state[np.newaxis, :], reward, terminal

    def _select_action(self, state: np.ndarray) -> int:
        """soon"""
        value = self._rng.random()
        if value < self.epsilon.value:
            action = int(self._rng.choice(self.env_actions))
        else:
            self._get_eval_model()
            action = self._evaluation_select_action(state)
        return action

    def _timestep(self, state: np.ndarray) -> Timestep:
        """soon"""
        action = self._select_action(state)
        next_state, reward, terminal = self._env_step(action)
        return Timestep(
            state,
            action,
            reward,
            next_state,
            terminal,
        )

    def _evaluation_select_action(self, state: np.ndarray) -> int:
        """soon"""
        values = self.eval_model(state).numpy()
        return int(np.argmax(values))

    def _evaluation_timestep(self, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """soon"""
        action = self._evaluation_select_action(state)
        return self._env_step(action)

    def _evaluation_episode(self) -> float:
        """soon"""
        sum_rewards = 0.0
        state = self.env.reset()[np.newaxis, :]
        for _ in range(self.eval.timesteps):
            state, reward, terminal = self._evaluation_timestep(state)
            sum_rewards += reward
            if terminal:
                break
        return sum_rewards

    def evaluation(self) -> float:
        """soon"""
        self._get_eval_model()
        rewards = np.zeros(self.eval.episodes)
        for t in range(self.eval.episodes):
            rewards[t] = self._evaluation_episode()
        return float(np.mean(rewards))

    @abstractmethod
    def _update_step(
        self,
        t: int,
        ts: Timestep,
    ) -> float:
        """soon"""
        pass

    def _episode(self, t: int) -> tuple[int, float, float]:
        """soon"""
        state = self.env.reset()[np.newaxis, :]
        reward = 0
        sum_error = 0.0
        for _t in range(self.max_timesteps):
            ts = self._timestep(state)
            sum_error += self._update_step(t, ts)
            reward += ts.reward
            t += 1
            state = ts.next_state
            if ts.terminal:
                break
        return t, sum_error / _t, reward

    def _experiment(
        self, experiment: int, episodes: int, threshold: int, pos: int
    ) -> int:
        """soon"""
        self.reset_models()
        t = 0
        for episode in (e_bar := tqdm(range(episodes), leave=False, position=pos)):
            self.epsilon.decay(episode)
            t, td_squared_error, reward = self._episode(t)
            self.rolling_mean.append(reward)
            self.rewards[experiment].append(reward)
            e_bar.set_postfix(
                reward=np.round(np.mean(self.rolling_mean), 2),  # type: ignore [call-overload]
                tde=td_squared_error,
            )
            self.td_errors[experiment].append(td_squared_error)
            if (episode + 1) % self.eval.frequency == 0:
                eval_reward = self.evaluation()
                self.mean_rewards[experiment].append(eval_reward)
                if eval_reward >= threshold:
                    break
        return episode + 1

    def run_experiment(
        self,
        n_experiments: int,
        episodes_per_experiment: int,
        threshold: int,
        pos: int = 0,
    ) -> np.ndarray:
        """soon"""
        self.steps_to_threshold = np.zeros(n_experiments)
        for e in (experiments := tqdm(range(n_experiments), position=pos)):
            experiments.set_description(f"{self.__class__.__name__} Experiment")
            self.mean_rewards.append([])
            self.rewards.append([])
            self.td_errors.append([])
            self.steps_to_threshold[e] = self._experiment(
                e, episodes_per_experiment, threshold, pos + 2
            )
        return self.steps_to_threshold

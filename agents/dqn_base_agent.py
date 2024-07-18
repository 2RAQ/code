from __future__ import annotations

from abc import abstractmethod
from collections import deque
from typing import TYPE_CHECKING

from keras.losses import Huber
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from environments.base_environment import Timestep

if TYPE_CHECKING:
    from gym import Env

    from agents.configurations import EvalConfig
    from agents.parameter import DecayParameter
    from environments.base_environment import RNG


class NNAgent:
    """Base class for agents using neural networks as function approximator instead
    of LFA via the theta parameterization.

    Since a random policy is very unlikely to visit all state action pairs, instead
    of the sample trajectory, the agents use epsilon-greedy exploration based on the
    current best estimator.

    Attributes:
        models: A list of n models similar to the theta in the LFA setting.
        eval_model: A model constructed form the set of models to be used for
            performance evaluation.
        alpha: The initial learning rate that will be used to initialize an Adam
            optimizer.
        gamma: The discount factor.
        epsilon: The epsilon parameter and its decay logic for the epsilon-greedy
            policy.
        optimizer: The optimizer used for the neural network training.
        huber_loss: The Huber loss function used for the neural network training.
        n_weights: The number of sets of parameters for the neural network.
        env: The environment used for the agent.
        env_actions: The action space of the environment.
        seed: The seed used for the random number generator.
        max_timesteps: The maximum number of timesteps per episode.
        mean_rewards: The mean rewards of episodes for each experiment.
        td_errors: The mean TD errors of episodes for each experiment.
        rolling_mean: Stores the mean rewards of the last 100 episodes.
    """

    models: list[tf.keras.models.Model]
    eval_model: tf.keras.models.Model

    def __init__(
        self,
        n_weights: int,
        alpha: float,
        epsilon: DecayParameter,
        gamma: float,
        env: Env,
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
        self.env: Env = env
        self.env_actions: np.ndarray = np.arange(4)
        self.eval: EvalConfig = eval_config
        self.seed: int = seed
        self.max_timesteps: int = max_steps
        self.mean_rewards: list[list[float]] = []
        self.rewards: list[list[float]] = []
        self.td_errors: list[list[float]] = []
        self.rolling_mean: deque[float] = deque(maxlen=100)

    def _init_models(self, n_weights: int) -> None:
        """Initialises a set of n copies of the model, as proxy for the number
        of thetas in the LFA setting.
        """
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
        """Resets all sets of parameters as well as the optimizer"""
        self._init_models(len(self.n_weights))
        self.optimizer = Adam(learning_rate=self.alpha)

    def update_model(
        self,
        model: tf.keras.models.Model,
        states: np.ndarray,
        targets: np.ndarray,
        training: bool = True,
    ) -> float:
        """Performs a gradient update based on the passed target values and returns
        the loss of the update.
        """
        with tf.GradientTape() as tape:
            value_predictions = model(states, training=training)
            loss = self.huber_loss(value_predictions, targets)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return float(loss)

    def _get_eval_model(self) -> None:
        """Generated an evaluation model based on the average of n models sets of
        weights. Approximates the method of taking the mean of multiple theta
        parameters. Returns
        """
        if len(self.models) == 1:
            self.eval_model.set_weights(self.models[0].get_weights())
            return None
        weights_list = [model.get_weights() for model in self.models]
        weights = np.array(weights_list, dtype=object)
        self.eval_model.set_weights(np.mean(weights, axis=0))

    def _env_step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Performs an action in the environment and returns the next state,
        the reward, and the termination signal.
        """
        next_state, reward, terminal, _ = self.env.step(action)  # type: ignore[misc]
        return next_state[np.newaxis, :], reward, terminal

    def _select_action(self, state: np.ndarray) -> int:
        """Selects an action based on the epsilon-greedy policy, where the greedy
        action selection depends on the used learning method of an agents instance.
        """
        value = self._rng.random()
        if value < self.epsilon.value:
            action = int(self._rng.choice(self.env_actions))
        else:
            self._get_eval_model()
            action = self._evaluation_select_action(state)
        return action

    def _timestep(self, state: np.ndarray) -> Timestep:
        """Performs a timestep in the environment and returns it as a Timestep
        object
        """
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
        """Selects a greedy action based on the Q-values of the evaluation model"""
        values = self.eval_model(state).numpy()
        return int(np.argmax(values))

    def _evaluation_timestep(self, state: np.ndarray) -> tuple[np.ndarray, float, bool]:
        action = self._evaluation_select_action(state)
        return self._env_step(action)

    def _evaluation_episode(self) -> float:
        """Runs one evaluation episode and returns the sum of rewards.
        The episode ends on reaching a terminal state of the environment.
        """
        sum_rewards = 0.0
        state = self.env.reset()[0][np.newaxis, :]
        for _ in range(self.eval.timesteps):
            state, reward, terminal = self._evaluation_timestep(state)
            sum_rewards += reward
            if terminal:
                break
        return sum_rewards

    def evaluation(self) -> float:
        """Evaluates the agent based on the evaluation configuration and returns the
        average reward over the evaluation episodes.
        """
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
        """Performs one update step based on the passed timestep and the current
        stepcount t. Returns the loss of the update.
        """

    def _episode(self, t: int) -> tuple[int, float, float]:
        """Runs one episode of the agent and returns the number of timesteps at
        which the episode was terminated, the mean squared error of the episode,
        and the accumulated reward.
        """
        state = self.env.reset()[0][np.newaxis, :]
        reward = 0.0
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
        """Runs one experiment in form of episodes number of episodes with the
        passed threshold as the success criterion. Performs evaluations in the
        frequency as specified in the eval configuration and returns the number
        of episodes it took to solve the environment.
        """
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

    def run_experiments(
        self,
        n_experiments: int,
        episodes_per_experiment: int,
        threshold: int,
        pos: int = 0,
    ) -> np.ndarray:
        """Runs n_experiments with n_repetitions each and n_samples samples per
        run.

        Args:
            n_experiments: The number of experiments to run.
            episodes_per_experiment: The number of episodes per experiment.
            threshold: The threshold at which an environment is considered solved
            pos: The position of the progress bar.
        """
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

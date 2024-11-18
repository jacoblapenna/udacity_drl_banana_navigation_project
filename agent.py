import copy
import random
import sqlite3
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from unityagents import UnityEnvironment

from model import QNetwork


DB_CONN = sqlite3.connect("training_results.db")
DB_CUR = DB_CONN.cursor()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DB_CUR.execute("CREATE TABLE IF NOT EXISTS training_results(gamma, tau, learning_rate, learning_frequency, scores, moving_averages)")


class TrainingScores:
    def __init__(self, window_size: int):
        self._scores = []
        self._moving_averages = []
        self._moving_average = 0.0
        self._length = 0
        self._window_size = window_size
        
    @property
    def scores(self):
        return self._scores
    
    @property
    def moving_average(self):
        return self._moving_average
    
    @property
    def moving_averages(self):
        return self._moving_averages
    
    @property
    def length(self):
        return self._length
    
    @property
    def window_size(self):
        return self._window_size
    
    def add_score(self, score):
        self._scores.append(score)
        self._length += 1
        
        if self._length > self._window_size:
            self._moving_average += (score - self._scores[-self._window_size]) / self._window_size
            self._moving_averages.append(self._moving_average)
        else:
            self._moving_average = (self._moving_average * (self._length - 1) + score) / self._length
            if self._length == self._window_size:
                self._moving_averages.append(self._moving_average)


class ExperienceBuffer:
    def __init__(self, buffer_size=int(1e5), batch_size=64):
        self._batch_size = batch_size
        self._buffer_size = buffer_size
        self._buffer = deque(maxlen=self._buffer_size)
        self._experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._length = 0
       
    @property
    def length(self):
        return self._length
    
    @property
    def batch_size(self):
        return self._batch_size
    
    def store(self, state, action, reward, next_state, done):
        self._buffer.append(self._experience(state, action, reward, next_state, done))
        if self._length < self._buffer_size:
            self._length += 1
        else:
            self._length = self._buffer_size
    
    def replay(self):
        if self._length >= self._batch_size:
            experiences = random.sample(self._buffer, k=self._batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)

            loaded_states = torch.from_numpy(np.vstack(states)).float().to(DEVICE)
            loaded_actions = torch.from_numpy(np.vstack(actions)).long().to(DEVICE)
            loaded_rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
            loaded_next_states = torch.from_numpy(np.vstack(next_states)).float().to(DEVICE)
            loaded_dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(DEVICE)
            
            return (loaded_states, loaded_actions, loaded_rewards, loaded_next_states, loaded_dones)
        else:
            raise Exception(f"Insufficient experiences to sample from. Length: {self._length}, Requested batch size: {self._batch_size}")


class Agent:    
    def __init__(
        self,
        env_path="/data/Banana_Linux_NoVis/Banana.x86_64",
        seed=0,
        learning_rate_alpha=5e-4,
        discount_rate_gamma=0.99,
        soft_update_tau=0.001,
        replay_buffer_size=int(1e5),
        learning_frequency=4
    ):
        """
        Params
        ======
            env_path (str): the path to the built environment
            seed (int): random seed, used for reproducability
            learning_rate (float): how softly to adjust the neural weights
            discount_rate_gamma (float): how much to value future states
            soft_update_tau (float): Controls how fast the target network converges to the local network
            replay_buffer_size (int): how many experiences to store in the buffer
            learning_frequency (int): the number of experiences to explore before cycling the learning step
        """
        self._alpha = learning_rate_alpha
        self._buffer_size = replay_buffer_size
        self._gamma = discount_rate_gamma
        self._tau = soft_update_tau
        self._seed = seed
        self._learning_frequency = learning_frequency

        self._env_path = env_path
        self._env = UnityEnvironment(file_name=self._env_path)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]

        self._action_size = self._brain.vector_action_space_size
        self._state_size = self._brain.vector_observation_space_size
        
        self._target_qnetwork = QNetwork(DEVICE, self._state_size, self._action_size).to(DEVICE)
        self._local_qnetwork = QNetwork(DEVICE, self._state_size, self._action_size).to(DEVICE)
        self._optimizer = optim.Adam(self._local_qnetwork.parameters(), lr=self._alpha)
        
        self._experience_buffer = ExperienceBuffer()
        self._training_scores = TrainingScores(window_size=100)
    
    @property
    def alpha(self):
        return self._alpha
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def random_seed(self):
        return self._seed
    
    @property
    def tau(self):
        return self._tau
    
    @property
    def training_scores(self):
        return self._training_scores
    
    def _live_plot(self, data: dict):
        pass
    
    def _explore(self, epsilon):
        """ Gain experiences """
        done = False
        score = 0
        episode_experience_count = 0
        observation = self._env.reset(train_mode=True)[self._brain_name]
        state = observation.vector_observations[0]
        
        self._local_qnetwork.eval()
        
        while not done:
            action = self._local_qnetwork.get_greedy_action_from_state(state, epsilon)
            observation = self._env.step(action)[self._brain_name]
            next_state = observation.vector_observations[0]
            reward = observation.rewards[0]
            done = observation.local_done[0]
            self._experience_buffer.store(state, action, reward, next_state, done)
            episode_experience_count += 1
            score += reward
            state = next_state
            if episode_experience_count % self._learning_frequency == 0:
                if self._experience_buffer.length > self._experience_buffer.batch_size:
                    self._learn()
        
        return score, episode_experience_count
        
    
    def _learn(self):
        """ Learn from experiences """
        states, actions, rewards, next_states, dones = self._experience_buffer.replay()
        
        next_targets= self._target_qnetwork(next_states).detach().max(1)[0].unsqueeze(1)
        target = rewards + (self._gamma * next_targets * (1 - dones))

        predicted = self._local_qnetwork(states).gather(1, actions)

        loss = F.mse_loss(predicted, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        self._soft_update()

    def _soft_update(self):
        with torch.no_grad():
            for target_param, local_param in zip(self._target_qnetwork.parameters(), self._local_qnetwork.parameters()):
                target_param.data.copy_(self._tau * local_param.data + (1.0 - self._tau) * target_param.data)
    
    def train_dqn(
        self,
        checkpoint_name,
        episodes=2000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay_factor=0.995,
        solution_score_threshold=13.0
    ):
        """ Run and train the agent under an epsilon greedy policy """        
        epsilon = epsilon_start

        for episode in range(1, episodes + 1):
            initial_norm = self._target_qnetwork.get_norm()
            score, episode_length = self._explore(epsilon)
            self._training_scores.add_score(score)
            epsilon = max(epsilon_min, epsilon * epsilon_decay_factor)
            final_norm = self._target_qnetwork.get_norm()

            print(
                f"\rEpisode {episode:05d} | "
                + f"Epsilon: {epsilon:.3f}"
                + f"\tLength: {episode_length}"
                + f"\tScore: {score:.2f}"
                + f"\tAverage Score: {self._training_scores.moving_average:.2f}"
                + f"\tPercent change: {self._target_qnetwork.percent_change(initial_norm, final_norm):.1E}",
                end=""
            )

            if self._training_scores.moving_average >= solution_score_threshold:
                print()
                print(f"\nEnvironment solved in {episode:d} episodes!\tAverage Score: {self._training_scores.moving_average:.2f}")
                
                torch.save(self._local_qnetwork.state_dict(), f"{checkpoint_name}.pth")
                DB_CUR.execute(
                    "INSERT INTO training_results VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        self._gamma,
                        self._tau,
                        self._alpha,
                        self._learning_frequency,
                        str(self.training_scores.scores),
                        str(self.training_scores.moving_averages)
                    )
                )
                DB_CONN.commit()
                
                return
    
    def close(self):
        self._env.close()
        DB_CONN.commit()
        DB_CONN.close()

        
class TrainedAgent:    
    def __init__(
        self,
        model_checkpoint_path,
        env_path="/data/Banana_Linux_NoVis/Banana.x86_64"
    ):
        """
        Params
        ======
            model_checkpoint_path (str): the path to the trained model checkpoint
            env_path (str): the path to the built environment
        """
        self._env_path = env_path
        self._env = UnityEnvironment(file_name=self._env_path)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]
    
        self._action_size = self._brain.vector_action_space_size
        self._state_size = self._brain.vector_observation_space_size
        
        self._model_checkpoint_path = model_checkpoint_path
        self._qnetwork = QNetwork(DEVICE, self._state_size, self._action_size).to(DEVICE)
        self._qnetwork.load_state_dict(torch.load(self._model_checkpoint_path))
     
    def run_model(self):
        done = False
        score = 0
        experience_count = 0
        observation = self._env.reset(train_mode=True)[self._brain_name]
        state = observation.vector_observations[0]
        
        self._local_qnetwork.eval()
        
        while not done:
            action = self._local_qnetwork.get_greedy_action_from_state(state, epsilon)
            observation = self._env.step(action)[self._brain_name]
            next_state = observation.vector_observations[0]
            reward = observation.rewards[0]
            done = observation.local_done[0]
            experience_count += 1
            score += reward
            state = next_state
            
            print(f"\rExperience count: {experience_count}\tScore: {score}", end="")
        
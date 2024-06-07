import gymnasium as gym
from tqdm import tqdm
import numpy as np
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from collections import deque

from replay_buffer import ReplayBuffer
from model import CNN
from environment import prepare_env

class DQN:
    def __init__(self, env:gym.Env, 
                 buffer_size=1000000, 
                 batch_size=128, #32 
                 gamma=0.99, 
                 lr=0.00025, 
                 update_freq=100, #4
                 target_update_freq=10000, 
                 eps_decay_steps=1e6, 
                 epsilon=0.1
                 ):
        self.env = env
        self.test_env = env
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.eps_decay_steps = eps_decay_steps
        self.epsilon = epsilon

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.q_network = CNN(4, self.env.action_space.n).to(self.device)
        self.target_network = CNN(4, self.env.action_space.n).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # self.criterion = torch.nn.MSELoss()
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)

        self.writer = SummaryWriter(f"runs/DQN_{self.env.spec.id}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")

    def get_epsilon(self, timestep, start_epsilon=1):
        if timestep < self.eps_decay_steps:
            # Linear decay
            epsilon = start_epsilon - (start_epsilon - self.epsilon) * (timestep / self.eps_decay_steps)
        else:
            # Constant value after decay period
            epsilon = self.epsilon
        return epsilon

    def select_action(self, state, info, test=False):
        epsilon = self.get_epsilon(timestep=info['frame_number'])
        if test:
            epsilon = 0.05
        if torch.rand(1) < epsilon:
            # Explore - Take a random action from the action space
            return np.random.choice(range(self.env.action_space.n))
        else:
            # Exploit - Follow the policy for the state
            state = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                Q = self.q_network(state)
            # max_indices = torch.where(Q == Q.max())[0]
            # return max_indices[torch.randint(0, len(max_indices), (1,))].item()
            return torch.argmax(Q).item()
  
    def optimize(self):
        # if len(self.replay_buffer) < self.batch_size:
        #     return
        
        # Sample random minibatch of transitions from D
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        # print(states.shape)
        # print(actions.shape)
        # print(rewards.shape)
        # print(next_states.shape)
        # print(dones.shape)

        q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_next = rewards + self.gamma * self.target_network(next_states).max(1)[0] * dones

        loss = self.criterion(q, q_next)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def test(self, frame_limit:int, writer:SummaryWriter, episode_number:int):
        state, info = self.test_env.reset()
        episode_reward = 0
        episode_steps = 0
        while True:
            episode_steps += 1
            action = self.select_action(state, info, test=True)
            next_state, reward, terminated, truncated, info = self.test_env.step(action)
            episode_reward += reward
            state = next_state
            if terminated or truncated or episode_steps >= frame_limit:
                break
                
        writer.add_scalar("Test/Episode Reward", episode_reward, episode_number)
        writer.add_scalar("Test/episode length (steps)", episode_steps, episode_number)

    def train(self, frame_limit:int):
        
        replay_start = 50000
        steps = 0
        frame_number = 0
        episode = 0
        episode_rewards = deque(maxlen=100)
        episode_losses = deque(maxlen=100)
        with tqdm(total=frame_limit, desc="Frames") as pbar:
            while frame_number < frame_limit:
                episode += 1
                state, info = self.env.reset()
                episode_loss = 0
                episode_reward = 0
                episode_steps = 0
                while True:
                    steps += 1
                    episode_steps += 1
                    
                    # for the first 30 episode steps, take a random action
                    # for the first 50 000 frames, take a random action to fill the replay buffer
                    if episode_steps < 30 or info['frame_number'] < replay_start:
                        action = self.env.action_space.sample()
                    else:
                        # With probability e select a random action a
                        action = self.select_action(state, info)
                    

                    # Execute action at in emulator and observe reward rt and image xt+1
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    frame_number = info['frame_number']
                    episode_reward += reward
                    done = True if terminated or truncated else False

                    # Store transition in replay buffer
                    self.replay_buffer.append((state, action, reward, next_state, done))

                    # Perform a gradient descent step
                    if steps % self.update_freq == 0 and info['frame_number'] >= replay_start:
                        loss = self.optimize()
                        episode_loss += loss.item()
                    
                    # Update current state
                    state = next_state
                    pbar.update(4)

                    if steps % self.target_update_freq == 0 and info['frame_number'] >= replay_start:
                        self.target_network.load_state_dict(self.q_network.state_dict())
                    
                    if terminated: # or truncated
                        break
                
                if episode % 100 == 0 and info['frame_number'] >= replay_start:
                    self.test(10000, self.writer, episode)
                
                episode_rewards.append(episode_reward)
                episode_losses.append(episode_loss)
                self.writer.add_scalar("epsilon", self.get_epsilon(info['frame_number']), info['frame_number'])
                self.writer.add_scalar("Train/episode length (steps)", episode_steps, episode)

                self.writer.add_scalar("Train/average episode reward - Last 100", np.mean(episode_rewards), episode)
                self.writer.add_scalar("Train/episode reward", episode_reward, episode)

                self.writer.add_scalar("Train/average episode loss - Last 100", np.mean(episode_losses), episode)
                self.writer.add_scalar("Train/episode loss", episode_loss, episode)
                
                if frame_number >= frame_limit:
                    break
        
        self.q_network.save(f"../models/dqn_update_delay_wtest.pth")

if __name__ == "__main__":
    env = prepare_env()
    dqn = DQN(env)
    dqn.train(10e6)
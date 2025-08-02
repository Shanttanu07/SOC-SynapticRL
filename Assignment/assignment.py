#!/usr/bin/env python3

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

np.random.seed(42)
torch.manual_seed(42)

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCE:
    def __init__(self, env_name='CartPole-v1', hidden_dim=128, lr=0.01, gamma=0.99):
        self.env = gym.make(env_name)
        self.gamma = gamma
        
        self.input_dim = self.env.observation_space.shape[0]
        self.output_dim = self.env.action_space.n
        
        self.policy = PolicyNetwork(self.input_dim, self.output_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.rewards = []
        self.action_probs = []
        self.episode_rewards = []
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state)
        action_probs = self.policy(state_tensor)
        distribution = Categorical(action_probs)
        action = distribution.sample()
        
        self.action_probs.append(distribution.log_prob(action))
        
        return action.item()
    
    def update_policy(self):
        if len(self.rewards) == 0:
            return
            
        returns = []
        R = 0
        
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        policy_loss = 0
        for log_prob, R in zip(self.action_probs, returns):
            policy_loss += -log_prob * R
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        self.rewards = []
        self.action_probs = []
    
    def train(self, num_episodes=1000, max_steps=1000, print_interval=100):
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state, _ = state
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.select_action(state)
                
                try:
                    next_state, reward, done, truncated, _ = self.env.step(action)
                except ValueError:
                    try:
                        next_state, reward, done, info = self.env.step(action)
                        truncated = False
                    except Exception as e:
                        print(f"Error in env.step: {e}")
                        done = True
                        truncated = True
                        break
                
                self.rewards.append(reward)
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    break
            
            self.episode_rewards.append(episode_reward)
            self.update_policy()
            
            if (episode + 1) % print_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-print_interval:])
                print(f"Episode {episode+1}/{num_episodes}, Average Reward: {avg_reward:.2f}")
        
        return self.episode_rewards
    
    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards)
        plt.title('Learning Curve: Policy Gradient (REINFORCE)')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        plt.savefig('policy_gradient_learning_curve.png')
        plt.show()
    
    def evaluate(self, num_episodes=10, render=False):
        total_rewards = []
        
        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state, _ = state
            episode_reward = 0
            done = False
            truncated = False
            
            while not (done or truncated):
                state_tensor = torch.FloatTensor(state)
                action_probs = self.policy(state_tensor)
                action = torch.argmax(action_probs).item()
                
                if render:
                    self.env.render()
                
                try:
                    next_state, reward, done, truncated, _ = self.env.step(action)
                except ValueError:
                    try:
                        next_state, reward, done, info = self.env.step(action)
                        truncated = False
                    except Exception as e:
                        print(f"Error in env.step: {e}")
                        done = True
                        truncated = True
                        break
                
                episode_reward += reward
                state = next_state
            
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        print(f"Average Evaluation Reward: {avg_reward:.2f}")
        return avg_reward

if __name__ == "__main__":
    agent = REINFORCE(env_name='CartPole-v1', hidden_dim=128, lr=0.01, gamma=0.99)
    rewards = agent.train(num_episodes=100, print_interval=10)
    
    agent.plot_learning_curve()
    agent.evaluate(num_episodes=5)

    print("Training and evaluation complete!")
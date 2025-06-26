import numpy as np
import gym
import random

# Load environment
env = gym.make("FrozenLake-v1", is_slippery=True)  # stochastic
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table initialization
q_table = np.zeros((n_states, n_actions))

# Hyperparameters
alpha = 0.8         # learning rate
gamma = 0.95        # discount factor
epsilon = 1.0       # initial exploration rate
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005
episodes = 2000
max_steps = 100

# Training
rewards_all_episodes = []

for ep in range(episodes):
    state = env.reset()[0]
    done = False
    total_rewards = 0

    for _ in range(max_steps):
        # Exploration vs Exploitation
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore
        else:
            action = np.argmax(q_table[state, :])  # exploit
        
        new_state, reward, done, truncated, _ = env.step(action)

        # Q-learning update
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action]
        )

        state = new_state
        total_rewards += reward

        if done or truncated:
            break

    # Decay epsilon
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)
    rewards_all_episodes.append(total_rewards)

# Evaluation
print("Q-table after training:\n", q_table)

# Test the trained policy
print("\nTesting the agent...")
test_episodes = 10
for ep in range(test_episodes):
    state = env.reset()[0]
    done = False
    print(f"\nEpisode {ep+1}:")
    for step in range(max_steps):
        env.render()
        action = np.argmax(q_table[state, :])
        new_state, reward, done, truncated, _ = env.step(action)
        state = new_state
        if done or truncated:
            env.render()
            if reward == 1:
                print("🎉 Reached Goal!")
            else:
                print("💥 Fell into a hole!")
            break

env.close()

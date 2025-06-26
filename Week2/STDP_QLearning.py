import numpy as np
import gym
import random

# Hyperparameters
alpha = 0.1           # Q-learning learning rate
gamma = 0.99          # Discount factor
epsilon = 1.0         # Exploration rate
decay_rate = 0.005
min_epsilon = 0.01
episodes = 1000
max_steps = 100

# STDP parameters
A_plus = 0.01
A_minus = 0.012
tau_plus = 20.0
tau_minus = 20.0

# Environment
env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n

# Q-table initialized using pseudo-synaptic weights
q_table = np.random.uniform(low=-1, high=1, size=(n_states, n_actions))
last_spike_time = np.full((n_states, n_actions), -np.inf)

# Time counter (simulating discrete spike events)
global_time = 0

def stdp_update(q_table, state, action, global_time):
    """
    Simulated STDP rule:
    - Presynaptic spike: state-action pair activation
    - Postsynaptic spike: action selected → interpreted as firing
    """
    t_pre = last_spike_time[state, action]
    t_post = global_time

    delta_t = t_post - t_pre

    if delta_t > 0:
        dw = A_plus * np.exp(-delta_t / tau_plus)
    else:
        dw = -A_minus * np.exp(delta_t / tau_minus)

    q_table[state, action] += dw
    last_spike_time[state, action] = global_time

for ep in range(episodes):
    state = env.reset()[0]
    done = False
    total_reward = 0

    for step in range(max_steps):
        global_time += 1

        # Epsilon-greedy
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state, :])

        # STDP learning (spike approximation)
        stdp_update(q_table, state, action, global_time)

        new_state, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Q-learning update
        q_table[state, action] += alpha * (reward + gamma * np.max(q_table[new_state, :]) - q_table[state, action])

        state = new_state
        if done:
            break

    # Decay epsilon
    epsilon = min_epsilon + (1.0 - min_epsilon) * np.exp(-decay_rate * ep)

    if ep % 100 == 0:
        print(f"Episode {ep}, Total Reward: {total_reward}")

# Final Q-table
print("\nFinal Q-table:\n", q_table)

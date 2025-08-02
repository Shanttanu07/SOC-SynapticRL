import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

class GridMaze:
    def __init__(self, size=5, obstacles=None):
        self.size = size
        self.grid = np.zeros((size, size))
        
        if obstacles is None:
            obstacles = []
            num_obstacles = size
            while len(obstacles) < num_obstacles:
                pos = (random.randint(0, size-1), random.randint(0, size-1))
                if pos != (0, 0) and pos != (size-1, size-1) and pos not in obstacles:
                    obstacles.append(pos)
        
        for obs in obstacles:
            self.grid[obs] = 1
            
        self.start = (0, 0)
        self.goal = (size-1, size-1)
        self.current_pos = self.start
        
    def reset(self):
        self.current_pos = self.start
        return self._get_state()
    
    def _get_state(self):
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def step(self, action):
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        new_pos = (self.current_pos[0] + directions[action][0], 
                  self.current_pos[1] + directions[action][1])
        
        if new_pos[0] < 0 or new_pos[0] >= self.size or new_pos[1] < 0 or new_pos[1] >= self.size:
            reward = -1
            done = False
            return self._get_state(), reward, done
        
        if self.grid[new_pos] == 1:
            reward = -1
            done = False
            return self._get_state(), reward, done
        
        self.current_pos = new_pos
        
        prev_distance = abs(self.start[0] - self.goal[0]) + abs(self.start[1] - self.goal[1])
        current_distance = abs(self.current_pos[0] - self.goal[0]) + abs(self.current_pos[1] - self.goal[1])
        
        if self.current_pos == self.goal:
            reward = 10
            done = True
        else:
            reward = -0.1
            
            if current_distance < prev_distance:
                reward += 0.5
                
            done = False
            
        return self._get_state(), reward, done
    
    def render(self, block=False, fig=None, ax=None):
        grid_viz = np.copy(self.grid)
        grid_viz[self.start] = 2
        grid_viz[self.goal] = 3
        grid_viz[self.current_pos] = 4
        
        if fig is None or ax is None:
            plt.close('maze')
            fig, ax = plt.subplots(figsize=(5, 5), num='maze')
        else:
            ax.clear()
            
        ax.imshow(grid_viz, cmap='Pastel1')
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.grid(True)
        ax.set_title('Grid Maze')
        fig.canvas.draw()
        plt.pause(0.1)

class SpikingNeuron:
    def __init__(self, threshold=1.0, leak=0.5):
        self.threshold = threshold
        self.leak = leak
        self.membrane_potential = 0.0
        self.last_spike_time = -1000
        
    def reset(self):
        self.membrane_potential = 0.0
        self.last_spike_time = -1000
        
    def forward(self, input_current, t):
        self.membrane_potential = self.leak * self.membrane_potential + input_current
        
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0.0
            self.last_spike_time = t
            return 1.0
        return 0.0

class STDPSynapse:
    def __init__(self, pre_neuron, post_neuron, weight=0.5, 
                 A_plus=0.1, A_minus=0.12, tau_plus=20.0, tau_minus=20.0):
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.weight = weight
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        
    def update(self, t):
        if self.pre_neuron.last_spike_time == t:
            dt = t - self.post_neuron.last_spike_time
            if dt > 0:
                self.weight -= self.A_minus * np.exp(-dt / self.tau_minus)
                
        if self.post_neuron.last_spike_time == t:
            dt = t - self.pre_neuron.last_spike_time
            if dt > 0:
                self.weight += self.A_plus * np.exp(-dt / self.tau_plus)
                
        self.weight = max(0.0, min(1.0, self.weight))
        
        return self.weight

class SNNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        self.input_neurons = [SpikingNeuron(threshold=0.9, leak=0.1) for _ in range(state_size)]
        self.output_neurons = [SpikingNeuron(threshold=1.0, leak=0.7) for _ in range(action_size)]
        
        self.synapses = {}
        for i in range(state_size):
            for j in range(action_size):
                self.synapses[(i, j)] = STDPSynapse(
                    self.input_neurons[i], 
                    self.output_neurons[j],
                    weight=np.random.uniform(0.4, 0.6)
                )
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen=1000)
        
    def reset(self):
        for neuron in self.input_neurons:
            neuron.reset()
        for neuron in self.output_neurons:
            neuron.reset()
            
    def choose_action(self, state, t):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        for neuron in self.input_neurons:
            neuron.reset()
            
        for i in range(self.state_size):
            if i == state:
                self.input_neurons[i].membrane_potential = 1.5
            else:
                self.input_neurons[i].membrane_potential = 0.0
                
        for i in range(self.state_size):
            self.input_neurons[i].forward(0, t)
            
        output_spikes = np.zeros(self.action_size)
        for j in range(self.action_size):
            input_current = 0
            for i in range(self.state_size):
                if self.input_neurons[i].last_spike_time == t:
                    input_current += self.synapses[(i, j)].weight
                    
            output_spikes[j] = self.output_neurons[j].forward(input_current, t)
        
        if np.sum(output_spikes) == 0:
            potentials = [n.membrane_potential for n in self.output_neurons]
            return np.argmax(potentials)
        
        return np.argmax(output_spikes)
    
    def remember(self, state, action, reward, next_state, done, t):
        self.memory.append((state, action, reward, next_state, done, t))
        
    def replay(self, batch_size, current_t):
        if len(self.memory) < batch_size:
            return
            
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done, t in minibatch:
            for neuron in self.input_neurons:
                neuron.reset()
            for neuron in self.output_neurons:
                neuron.reset()
                
            for i in range(self.state_size):
                if i == state:
                    self.input_neurons[i].membrane_potential = 1.5
                    spike = self.input_neurons[i].forward(0, current_t)
            
            scaling_factor = 1.0 + reward
            
            input_current = 0
            for i in range(self.state_size):
                if self.input_neurons[i].last_spike_time == current_t:
                    input_current += self.synapses[(i, action)].weight * scaling_factor
            
            self.output_neurons[action].forward(input_current, current_t)
            
            for i in range(self.state_size):
                for j in range(self.action_size):
                    self.synapses[(i, j)].update(current_t)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent(episodes=1000, render_interval=100):
    plt.ion()
    
    maze = GridMaze(size=5)
    state_size = maze.size * maze.size
    action_size = 4
    
    agent = SNNAgent(state_size, action_size)
    
    batch_size = 32
    max_steps = 100
    
    rewards_history = []
    steps_history = []
    success_history = []
    
    metrics_fig, metrics_axes = plt.subplots(1, 3, figsize=(15, 5), num='metrics')
    maze_fig, maze_ax = plt.subplots(figsize=(5, 5), num='maze')
    
    for episode in range(episodes):
        state = maze.reset()
        agent.reset()
        total_reward = 0
        steps = 0
        
        for t in range(max_steps):
            action = agent.choose_action(state, t)
            next_state, reward, done = maze.step(action)
            
            agent.remember(state, action, reward, next_state, done, t)
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                success_history.append(1)
                break
                
        if not done:
            success_history.append(0)
            
        agent.replay(batch_size, episode)
        rewards_history.append(total_reward)
        steps_history.append(steps)
        
        if episode % render_interval == 0:
            print(f"Episode: {episode}, Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {agent.epsilon:.2f}")
            if episode > 0:
                for ax in metrics_axes:
                    ax.clear()
                
                metrics_axes[0].plot(rewards_history)
                metrics_axes[0].set_xlabel('Episode')
                metrics_axes[0].set_ylabel('Total Reward')
                metrics_axes[0].set_title('Rewards History')
                
                metrics_axes[1].plot(steps_history)
                metrics_axes[1].set_xlabel('Episode')
                metrics_axes[1].set_ylabel('Steps')
                metrics_axes[1].set_title('Steps per Episode')
                
                window_size = min(100, len(success_history))
                success_rate = [sum(success_history[max(0, i-window_size):i])/window_size 
                               if i >= window_size else sum(success_history[:i])/i 
                               if i > 0 else 0 
                               for i in range(1, len(success_history)+1)]
                metrics_axes[2].plot(success_rate)
                metrics_axes[2].set_xlabel('Episode')
                metrics_axes[2].set_ylabel('Success Rate')
                metrics_axes[2].set_title(f'Success Rate (Window: {window_size})')
                
                metrics_fig.tight_layout()
                metrics_fig.canvas.draw()
                plt.pause(0.1)
            
            maze.render(block=False, fig=maze_fig, ax=maze_ax)
            
    plt.ioff()
    return agent, maze, rewards_history, steps_history, success_history

if __name__ == "__main__":
    import time
    
    print("Training the agent...")
    agent, maze, rewards, steps, success_rate = train_agent(episodes=500, render_interval=50)
    
    print("\nTesting the trained agent...")
    plt.ion()
    
    test_fig, test_ax = plt.subplots(figsize=(5, 5), num='test_maze')
    
    state = maze.reset()
    done = False
    total_reward = 0
    step_count = 0
    
    while not done and step_count < 100:
        action = agent.choose_action(state, 0)
        state, reward, done = maze.step(action)
        total_reward += reward
        step_count += 1
        
        maze.render(block=False, fig=test_fig, ax=test_ax)
        time.sleep(0.3)
        
    plt.ioff()
    
    if done:
        print(f"\nAgent successfully reached the goal in {step_count} steps with total reward: {total_reward:.2f}")
    else:
        print(f"\nAgent failed to reach the goal. Total steps: {step_count}, Total reward: {total_reward:.2f}")
    
    maze.render(block=True, fig=test_fig, ax=test_ax)
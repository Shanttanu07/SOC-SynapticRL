"""
Implementation of a standard Q-Learning agent in the same maze environment
for comparison with the MAML approach.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import time
import sys
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


class MazeEnvironment:
    """Maze environment with configurable layout."""
    
    def __init__(self, size=5, random_maze=False, maze_layout=None):
        self.size = size
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left
        
        if maze_layout is not None:
            self.maze = np.array(maze_layout)
            self.size = self.maze.shape[0]
        elif random_maze:
            self.maze = self._generate_random_maze()
        else:
            self.maze = np.array([
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0]
            ])
        
        self.start_pos = (0, 0)
        self.goal_pos = (self.size - 1, self.size - 1)
        self.agent_pos = self.start_pos
        self.action_space = 4
        self.max_steps = self.size * self.size * 3
        self.steps = 0
    
    def _generate_random_maze(self):
        """Generate a random maze with guaranteed path from start to goal."""
        maze = np.ones((self.size, self.size))
        
        maze[0, 0] = 0
        maze[self.size-1, self.size-1] = 0
        
        current = (0, 0)
        visited = [current]
        
        while current != (self.size-1, self.size-1):
            x, y = current
            neighbors = []
            
            for dx, dy in self.actions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    neighbors.append((nx, ny))
            
            if neighbors:
                next_pos = random.choice(neighbors)
                maze[next_pos] = 0
                current = next_pos
                visited.append(current)
            else:
                visited.pop()
                if visited:
                    current = visited[-1]
                else:
                    break
        
        for _ in range(self.size * self.size // 2):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (x, y) != (0, 0) and (x, y) != (self.size-1, self.size-1):
                maze[x, y] = random.choice([0, 1])
        
        return maze
    
    def reset(self):
        """Reset environment to initial state."""
        self.agent_pos = self.start_pos
        self.steps = 0
        return self._get_state()
    
    def _get_state(self):
        """Return current state representation."""
        x, y = self.agent_pos
        state = np.zeros(2 + 4)
        
        state[0] = x / (self.size - 1)
        state[1] = y / (self.size - 1)
        
        for i, (dx, dy) in enumerate(self.actions):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                state[2 + i] = self.maze[nx, ny]
            else:
                state[2 + i] = 1  # Treat boundaries as walls
        
        return state
    
    def step(self, action):
        """Take a step in the environment."""
        dx, dy = self.actions[action]
        x, y = self.agent_pos
        
        new_x, new_y = x + dx, y + dy
        
        if (0 <= new_x < self.size and 
            0 <= new_y < self.size and 
            self.maze[new_x, new_y] == 0):
            self.agent_pos = (new_x, new_y)
        
        self.steps += 1
        
        if done:
            reward = 10.0  # Goal reached
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0  # Max steps reached
            done = True
        else:
            # Calculate distance-based reward
            old_dist = abs(x - self.goal_pos[0]) + abs(y - self.goal_pos[1])
            new_dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 + 0.1 * (old_dist - new_dist)
            done = False
        
        return self._get_state(), reward, done
    
    def render(self):
        """Render the maze environment."""
        rendered_maze = np.copy(self.maze).astype(str)
        rendered_maze[rendered_maze == '0'] = '.'
        rendered_maze[rendered_maze == '1'] = '#'
        
        rendered_maze[self.start_pos] = 'S'
        rendered_maze[self.goal_pos] = 'G'
        
        if self.agent_pos != self.start_pos and self.agent_pos != self.goal_pos:
            rendered_maze[self.agent_pos] = 'A'
            
        for row in rendered_maze:
            print(' '.join(row))
        print()
    
    def create_variations(self, num_variations):
        """Generate variations of the maze for meta-learning."""
        variations = []
        for _ in range(num_variations):
            new_maze = MazeEnvironment(self.size, random_maze=True)
            variations.append(new_maze)
        return variations


class QLearningAgent:
    """Standard Q-Learning Agent for maze navigation."""
    
    def __init__(self, state_size, action_size, discretize=True, learning_rate=0.1, 
                 discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, 
                 min_exploration_rate=0.01):
        self.action_size = action_size
        self.discretize = discretize
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration_rate = min_exploration_rate
        
        if discretize:
            # Use a dictionary for discretized state space
            self.q_table = {}
        else:
            # Use a numpy array for fixed-size state space
            self.state_size = 5 * 5 * 2**4
            self.q_table = np.zeros((self.state_size, action_size))
    
    def discretize_state(self, state):
        """Convert continuous state to discrete representation for table lookup."""
        if self.discretize:
            # Extract position and surroundings
            x, y = state[0], state[1]
            surroundings = tuple(int(s) for s in state[2:])
            
            # Discretize position to integer coordinates (0-4)
            x_discrete = min(int(x * 5), 4)
            y_discrete = min(int(y * 5), 4)
            
            return (x_discrete, y_discrete, surroundings)
        else:
            # Convert state to an index for the Q-table
            x, y = state[0], state[1]
            surroundings = state[2:]
            
            x_discrete = min(int(x * 5), 4)
            y_discrete = min(int(y * 5), 4)
            
            # Encode surroundings as a binary number
            surroundings_code = 0
            for i, wall in enumerate(surroundings):
                if wall > 0.5:
                    surroundings_code += 2**i
            
            # Combine position and surroundings into a single index
            return x_discrete * 5 * 16 + y_discrete * 16 + surroundings_code
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        if self.discretize:
            state_key = self.discretize_state(state)
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
            return self.q_table[state_key][action]
        else:
            state_idx = self.discretize_state(state)
            return self.q_table[state_idx, action]
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using the Q-learning update rule."""
        if self.discretize:
            state_key = self.discretize_state(state)
            next_state_key = self.discretize_state(next_state)
            
            if state_key not in self.q_table:
                self.q_table[state_key] = np.zeros(self.action_size)
                
            if next_state_key not in self.q_table:
                self.q_table[next_state_key] = np.zeros(self.action_size)
            
            current_q = self.q_table[state_key][action]
            
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.q_table[next_state_key])
                target_q = reward + self.discount_factor * max_next_q
            
            self.q_table[state_key][action] += self.learning_rate * (target_q - current_q)
        else:
            state_idx = self.discretize_state(state)
            next_state_idx = self.discretize_state(next_state)
            
            current_q = self.q_table[state_idx, action]
            
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.q_table[next_state_idx])
                target_q = reward + self.discount_factor * max_next_q
            
            self.q_table[state_idx, action] += self.learning_rate * (target_q - current_q)
    
    def choose_action(self, state):
        """Choose an action using epsilon-greedy policy."""
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_size - 1)
        else:
            if self.discretize:
                state_key = self.discretize_state(state)
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.action_size)
                return np.argmax(self.q_table[state_key])
            else:
                state_idx = self.discretize_state(state)
                return np.argmax(self.q_table[state_idx])
    
    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate, 
                                   self.exploration_rate * self.exploration_decay)


def train_q_learning_agent(env, episodes=500, max_steps=100, render_interval=100):
    """Train a Q-learning agent on a specific maze environment."""
    agent = QLearningAgent(state_size=6, action_size=4)
    
    rewards_history = []
    success_history = []
    steps_history = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            if episode % render_interval == 0 and steps == 0:
                print(f"Episode {episode}, Exploration rate: {agent.exploration_rate:.4f}")
                env.render()
            
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            agent.update_q_value(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
        
        agent.decay_exploration()
        rewards_history.append(total_reward)
        success = done and env.agent_pos == env.goal_pos
        success_history.append(1 if success else 0)
        steps_history.append(steps)
        
        if episode % render_interval == 0:
            success_rate = sum(success_history[-100:]) / min(len(success_history), 100)
            avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)
            avg_steps = sum(steps_history[-100:]) / min(len(steps_history), 100)
            print(f"Episode {episode}: Reward={total_reward:.2f}, Steps={steps}, Success={success}")
            print(f"Last 100 episodes: Success rate={success_rate:.2f}, Avg reward={avg_reward:.2f}, Avg steps={avg_steps:.2f}")
            if success:
                print("Goal reached!")
                env.render()
            print()
    
    # Calculate final statistics
    final_success_rate = sum(success_history[-100:]) / min(len(success_history), 100)
    final_avg_reward = sum(rewards_history[-100:]) / min(len(rewards_history), 100)
    final_avg_steps = sum(steps_history[-100:]) / min(len(steps_history), 100)
    
    print(f"Training completed: Final success rate={final_success_rate:.2f}, Avg reward={final_avg_reward:.2f}, Avg steps={final_avg_steps:.2f}")
    
    return agent, rewards_history, success_history, steps_history


def evaluate_agent(env, agent, num_episodes=10, render=True):
    """Evaluate a trained agent on a specific environment."""
    total_rewards = []
    success_count = 0
    steps_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        if render and episode == 0:
            print("\nEvaluation - Initial state:")
            env.render()
        
        while not done and steps < 100:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            
            if render and episode == 0 and steps % 5 == 0:
                print(f"Step {steps}, Action: {action}")
                env.render()
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done and env.agent_pos == env.goal_pos:
                success_count += 1
        
        if render and episode == 0:
            print(f"Episode finished after {steps} steps. Goal reached: {env.agent_pos == env.goal_pos}")
            env.render()
        
        total_rewards.append(total_reward)
        steps_list.append(steps)
    
    avg_reward = sum(total_rewards) / num_episodes
    success_rate = success_count / num_episodes
    avg_steps = sum(steps_list) / num_episodes
    
    print(f"Evaluation results:")
    print(f"  Average Reward: {avg_reward:.2f}")
    print(f"  Success Rate: {success_rate:.2f}")
    print(f"  Average Steps: {avg_steps:.2f}")
    
    return avg_reward, success_rate, avg_steps


def compare_approaches():
    """Compare Q-Learning approach on maze environments."""
    print("=" * 50)
    print("Q-LEARNING ON MAZE NAVIGATION")
    print("=" * 50)
    
    # Create base maze environment
    base_maze = MazeEnvironment(size=5)
    
    # Create a set of test mazes for evaluation
    num_test_mazes = 5
    test_mazes = base_maze.create_variations(num_test_mazes)
    
    # Timer for Q-Learning training
    q_learning_start_time = time.time()
    
    # Train Q-Learning agents on each test maze
    print("\nTraining Q-Learning agents on test mazes...")
    q_learning_agents = []
    q_learning_results = []
    
    for i, maze in enumerate(test_mazes):
        print(f"\nTraining Q-Learning agent on Test Maze {i+1}:")
        agent, rewards, successes, steps = train_q_learning_agent(maze, episodes=500, render_interval=100)
        q_learning_agents.append(agent)
        
        # Evaluate the trained agent
        print(f"\nEvaluating Q-Learning agent on Test Maze {i+1}:")
        avg_reward, success_rate, avg_steps = evaluate_agent(maze, agent, num_episodes=10, render=True)
        q_learning_results.append((avg_reward, success_rate, avg_steps))
        
        # Plot learning curve
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(rewards)
        plt.title(f'Q-Learning Rewards - Maze {i+1}')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot([sum(successes[max(0, i-100):i+1]) / min(i+1, 100) for i in range(len(successes))])
        plt.title(f'Q-Learning Success Rate - Maze {i+1}')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate (moving avg)')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(steps)
        plt.title(f'Q-Learning Steps - Maze {i+1}')
        plt.xlabel('Episode')
        plt.ylabel('Steps per Episode')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'q_learning_maze_{i+1}.png')
        plt.close()
    
    q_learning_time = time.time() - q_learning_start_time
    
    # Summary of Q-Learning performance
    q_learning_avg_success = sum(result[1] for result in q_learning_results) / num_test_mazes
    
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"\nQ-Learning Training Time: {q_learning_time:.2f} seconds")
    print(f"Q-Learning Average Success Rate: {q_learning_avg_success:.2f}")
    
    print("\nPerformance on test mazes:")
    for i in range(num_test_mazes):
        print(f"\nTest Maze {i+1}:")
        print(f"  Q-Learning: Avg Reward={q_learning_results[i][0]:.2f}, Success Rate={q_learning_results[i][1]:.2f}, Avg Steps={q_learning_results[i][2]:.2f}")
    
    return q_learning_agents, test_mazes


def train_on_new_maze(q_agents, test_mazes):
    """Train a Q-Learning agent on a new, larger maze."""
    print("\nAdditional experiment: Training Q-Learning on a new maze...")
    new_maze = MazeEnvironment(size=7, random_maze=True)
    
    print("\nTraining Q-Learning agent on new maze:")
    q_agent, _, _, _ = train_q_learning_agent(new_maze, episodes=1000, render_interval=200)
    
    print("\nEvaluating trained Q-Learning agent on new maze:")
    q_avg_reward, q_success_rate, q_avg_steps = evaluate_agent(new_maze, q_agent, num_episodes=10, render=True)


if __name__ == "__main__":
    print("Starting Q-Learning training and evaluation...")
    q_agents, test_environments = compare_approaches()
    train_on_new_maze(q_agents, test_environments)
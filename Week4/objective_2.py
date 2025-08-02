"""
Implementation of a meta-learning agent using MAML (Model-Agnostic Meta-Learning) 
in a maze environment.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import copy

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MazeEnvironment:
    """Maze environment with configurable layout."""
    
    def __init__(self, size=5, random_maze=False, maze_layout=None):
        self.size = size
        
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
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
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
        
        if self.agent_pos == self.goal_pos:
            reward = 10.0  # Reached goal
            done = True
        elif self.steps >= self.max_steps:
            reward = -1.0  # Took too many steps
            done = True
        else:
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


class PolicyNetwork(nn.Module):
    """Neural network policy for the agent."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        probs = self.forward(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[0][action].item()


class MAML:
    """Model-Agnostic Meta-Learning implementation for maze navigation."""
    
    def __init__(self, input_size, hidden_size, output_size, alpha=0.1, beta=0.01):
        self.model = PolicyNetwork(input_size, hidden_size, output_size).to(device)
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=beta)
        self.alpha = alpha  # Inner loop learning rate
        self.beta = beta    # Outer loop learning rate
    
    def adapt(self, task_env, num_steps=5):
        """Adapt to a specific task (inner loop of MAML)."""
        adapted_model = copy.deepcopy(self.model)
        adapted_model.train()
        
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.alpha)
        
        for _ in range(num_steps):
            states, actions, rewards = self._collect_episode(task_env, adapted_model)
            returns = self._calculate_returns(rewards)
            loss = self._calculate_loss(adapted_model, states, actions, returns)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return adapted_model
    
    def meta_update(self, task_envs, num_adapt_steps=5, num_meta_steps=1):
        """Perform meta-update (outer loop of MAML)."""
        self.model.train()
        meta_loss = 0.0
        
        for _ in range(num_meta_steps):
            tasks_batch = random.sample(task_envs, min(len(task_envs), 5))
            
            batch_loss = 0.0
            for task_env in tasks_batch:
                adapted_model = self.adapt(task_env, num_adapt_steps)
                
                states, actions, rewards = self._collect_episode(task_env, adapted_model)
                returns = self._calculate_returns(rewards)
                
                task_loss = self._calculate_loss(adapted_model, states, actions, returns)
                batch_loss += task_loss
            
            batch_loss /= len(tasks_batch)
            meta_loss += batch_loss.item()
            
            self.meta_optimizer.zero_grad()
            batch_loss.backward()
            self.meta_optimizer.step()
        
        return meta_loss / num_meta_steps
    
    def _collect_episode(self, env, model, max_steps=100):
        """Collect a single episode using the given model."""
        states = []
        actions = []
        rewards = []
        
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < max_steps:
            action, _ = model.get_action(state)
            next_state, reward, done = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            step += 1
        
        return torch.FloatTensor(states).to(device), torch.LongTensor(actions).to(device), rewards
    
    def _calculate_returns(self, rewards, gamma=0.99):
        """Calculate discounted returns."""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(device)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def _calculate_loss(self, model, states, actions, returns):
        """Calculate policy loss."""
        probs = model(states)
        action_probs = probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        log_probs = torch.log(action_probs + 1e-8)
        loss = -torch.mean(log_probs * returns)
        
        return loss
    
    def evaluate(self, env, num_episodes=10, render=False):
        """Evaluate the meta-model on a specific environment."""
        adapted_model = self.adapt(env)
        
        total_rewards = []
        success_count = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                if render:
                    env.render()
                
                action, _ = adapted_model.get_action(state)
                next_state, reward, done = env.step(action)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done and env.agent_pos == env.goal_pos:
                    success_count += 1
            
            total_rewards.append(total_reward)
        
        avg_reward = sum(total_rewards) / num_episodes
        success_rate = success_count / num_episodes
        
        return avg_reward, success_rate, adapted_model


def train_meta_learning_agent():
    """Train a meta-learning agent on a set of maze tasks."""
    base_maze = MazeEnvironment(size=5)
    
    num_variations = 20
    maze_variations = base_maze.create_variations(num_variations)
    
    input_size = 6  # state size (position + surroundings)
    hidden_size = 64
    output_size = 4  # action space size
    maml = MAML(input_size, hidden_size, output_size, alpha=0.1, beta=0.01)
    
    print("Starting meta-training...")
    num_meta_epochs = 50
    meta_losses = []
    
    for epoch in range(num_meta_epochs):
        meta_loss = maml.meta_update(maze_variations, num_adapt_steps=3, num_meta_steps=1)
        meta_losses.append(meta_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"Meta-Epoch {epoch+1}/{num_meta_epochs}, Meta-Loss: {meta_loss:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(meta_losses)
    plt.title('Meta-Learning Loss')
    plt.xlabel('Meta-Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('meta_learning_curve.png')
    
    print("\nEvaluating on test mazes...")
    test_mazes = base_maze.create_variations(5)
    
    for i, test_maze in enumerate(test_mazes):
        pre_reward_total = 0
        pre_success_total = 0
        
        for _ in range(10):  # Pre-adaptation episodes
            state = test_maze.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done and steps < 100:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                probs = maml.model(state_tensor)
                action = torch.multinomial(probs, 1).item()
                
                next_state, reward, done = test_maze.step(action)
                total_reward += reward
                state = next_state
                steps += 1
                
                if done and test_maze.agent_pos == test_maze.goal_pos:
                    pre_success_total += 1
            
            pre_reward_total += total_reward
        
        pre_avg_reward = pre_reward_total / 10
        pre_success_rate = pre_success_total / 10
        
        post_avg_reward, post_success_rate, adapted_model = maml.evaluate(test_maze, num_episodes=10)
        
        print(f"Test Maze {i+1}:")
        print(f"  Pre-adaptation:  Avg Reward: {pre_avg_reward:.2f}, Success Rate: {pre_success_rate:.2f}")
        print(f"  Post-adaptation: Avg Reward: {post_avg_reward:.2f}, Success Rate: {post_success_rate:.2f}")
        
        print("\nVisualizing one episode with adapted policy:")
        state = test_maze.reset()
        done = False
        steps = 0
        
        print("Initial maze state:")
        test_maze.render()
        
        while not done and steps < 50:
            action, _ = adapted_model.get_action(state)
            next_state, reward, done = test_maze.step(action)
            
            if steps % 5 == 0:  # Show every 5 steps
                print(f"Step {steps}, Action: {action}")
                test_maze.render()
            
            state = next_state
            steps += 1
        
        if done:
            print(f"Episode finished after {steps} steps. Goal reached: {test_maze.agent_pos == test_maze.goal_pos}")
            test_maze.render()
    
    return maml

if __name__ == "__main__":
    maml_agent = train_meta_learning_agent()
    
    print("\nFinal demonstration on a new maze:")
    final_maze = MazeEnvironment(size=7, random_maze=True)
    
    _, final_success_rate, _ = maml_agent.evaluate(final_maze, num_episodes=10, render=True)
    print(f"Final test maze success rate: {final_success_rate:.2f}")
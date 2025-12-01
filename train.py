"""
GPU-only TD3 trainer for CartPole (single-file)

Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation.
Requires CUDA-capable GPU. Trains with multiple seeds.

Usage:
  python train.py --episodes 300
"""

import argparse
import random
from collections import deque
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3Agent:
    def __init__(self, state_dim, action_dim, max_action, device,
                 lr=3e-4, gamma=0.99, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=2, batch_size=256):
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay = ReplayBuffer()
        self.total_it = 0

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        if noise > 0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
        return action.clip(-self.max_action, self.max_action)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return None, None

        self.total_it += 1

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)

        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            actor_loss = actor_loss.item()

        return critic_loss.item(), actor_loss

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])


def train(env_name, episodes, device, seed=0, max_steps=1000):
    # Set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action, device)

    episode_rewards = []
    exploration_noise = 0.1

    print(f"\n{'='*70}")
    print(f"Training with seed {seed}")
    print(f"{'='*70}")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_steps):
            action = agent.select_action(state, noise=exploration_noise)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay.push(state, action, reward, next_state, float(done))
            critic_loss, actor_loss = agent.train_step()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if ep % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"  Episode {ep:4d}/{episodes} | Reward: {episode_reward:7.2f} | Avg(10): {avg_reward:7.2f} | Updates: {agent.total_it:6d}")

    env.close()
    
    print(f"\nTraining completed! Total updates: {agent.total_it}")
    print(f"Final 10 episodes avg: {np.mean(episode_rewards[-10:]):.2f}")
    
    return agent, episode_rewards


def evaluate(agent, env_name, seed=10, episodes=10):
    env = gym.make(env_name)
    env.reset(seed=seed)
    
    episode_rewards = []
    
    print(f"\n{'='*70}")
    print(f"Evaluating with seed {seed}")
    print(f"{'='*70}")
    
    for ep in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        print(f"  Episode {ep + 1}/{episodes}: Reward = {episode_reward:.2f}")
    
    env.close()
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    
    print(f"\nEvaluation Results:")
    print(f"  Mean: {mean_reward:.2f}")
    print(f"  Std: {std_reward:.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    
    return episode_rewards


def plot_learning_curves(all_rewards, eval_rewards, save_path='results/learning_curve.png'):
    all_rewards = np.array(all_rewards)
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)
    
    plt.figure(figsize=(12, 6))
    episodes = np.arange(1, len(mean_rewards) + 1)
    
    plt.plot(episodes, mean_rewards, label='Training Mean', color='#2E86AB', linewidth=2)
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     alpha=0.3, color='#2E86AB', label='Training Std')
    
    eval_mean = np.mean(eval_rewards)
    eval_std = np.std(eval_rewards)
    plt.axhline(y=eval_mean, color='#A23B72', linestyle='--', linewidth=2, label='Evaluation Mean')
    plt.fill_between([0, len(mean_rewards)], eval_mean - eval_std, eval_mean + eval_std,
                     alpha=0.2, color='#A23B72')
    
    plt.xlabel('Episode', fontsize=13, fontweight='bold')
    plt.ylabel('Reward', fontsize=13, fontweight='bold')
    plt.title('TD3 Learning Curve - InvertedPendulum-v4', fontsize=15, fontweight='bold', pad=15)
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nLearning curve saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train TD3 on InvertedPendulum-v4')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--eval-episodes', type=int, default=10)
    parser.add_argument('--env', type=str, default='InvertedPendulum-v4')
    args = parser.parse_args()

    # Enforce GPU-only execution
    if not torch.cuda.is_available():
        raise RuntimeError(
            "ERROR: CUDA is not available. This script requires GPU execution.\n"
            "Please check:\n"
            "  1. GPU drivers are installed: nvidia-smi\n"
            "  2. PyTorch with CUDA support is installed\n"
            "  3. CUDA is properly configured"
        )

    device = torch.device('cuda')

    print("\n" + "="*70)
    print("TD3 TRAINING - GPU ONLY")
    print("="*70)
    print(f"Environment: {args.env}")
    print(f"Training Episodes: {args.episodes}")
    print(f"Evaluation Episodes: {args.eval_episodes}")
    print(f"Training Seeds: [0, 1, 2]")
    print(f"Evaluation Seed: 10")
    print(f"\nDevice: GPU (CUDA)")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print("="*70)

    # Train with multiple seeds
    TRAIN_SEEDS = [0, 1, 2]
    EVAL_SEED = 10
    
    all_training_rewards = []
    all_agents = []
    all_eval_rewards = []

    for seed in TRAIN_SEEDS:
        agent, rewards = train(args.env, args.episodes, device, seed=seed)
        all_training_rewards.append(rewards)
        all_agents.append(agent)

    # Evaluate all agents
    for i, agent in enumerate(all_agents):
        print(f"\nEvaluating agent trained with seed {TRAIN_SEEDS[i]}...")
        eval_rewards = evaluate(agent, args.env, seed=EVAL_SEED, episodes=args.eval_episodes)
        all_eval_rewards.append(np.mean(eval_rewards))

    # Print summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    print("\nTraining Performance (last 10 episodes):")
    for i, seed in enumerate(TRAIN_SEEDS):
        final_mean = np.mean(all_training_rewards[i][-10:])
        print(f"  Seed {seed}: {final_mean:.2f}")
    
    print(f"\nEvaluation Performance (seed={EVAL_SEED}):")
    for i, seed in enumerate(TRAIN_SEEDS):
        print(f"  Agent (seed {seed}): {all_eval_rewards[i]:.2f}")
    
    overall_eval_mean = np.mean(all_eval_rewards)
    overall_eval_std = np.std(all_eval_rewards)
    print(f"\nOverall Evaluation: {overall_eval_mean:.2f} ± {overall_eval_std:.2f}")
    print("="*70)

    # Save results
    os.makedirs('results', exist_ok=True)
    plot_learning_curves(all_training_rewards, all_eval_rewards)
    
    # Save best model
    best_idx = np.argmax(all_eval_rewards)
    best_agent = all_agents[best_idx]
    best_seed = TRAIN_SEEDS[best_idx]
    
    best_agent.save('results/best_model.pt')
    print(f"\nBest agent (trained with seed {best_seed}) saved to results/best_model.pt")
    print(f"Evaluation reward: {all_eval_rewards[best_idx]:.2f}")
    
    print(f"\nAll results saved to 'results/' directory")
    print("\nTraining completed successfully!")


if __name__ == '__main__':
    main()

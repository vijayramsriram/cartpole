"""
Visualize trained TD3 agent balancing pole indefinitely in real-time
"""
import argparse
import torch
import gymnasium as gym
import numpy as np
import time
from train import TD3Agent


def visualize(model_path='results/best_model.pt', render=True, continuous=False):
    env_name = 'InvertedPendulum-v4'
    
    # Create environment with rendering
    if render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)
    
    # Initialize agent
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = TD3Agent(state_dim, action_dim, max_action, device)
    
    # Load trained model
    print(f"Loading model from {model_path}...")
    agent.load(model_path)
    print("Model loaded successfully!\n")
    
    if continuous:
        print("Running in CONTINUOUS mode - balancing indefinitely")
        print("Press Ctrl+C to stop\n")
        
        try:
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            episode_count = 1
            
            while True:
                # Select action (no exploration noise)
                action = agent.select_action(state, noise=0.0)
                
                # Step environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                steps += 1
                
                # Slow down for better viewing
                time.sleep(0.02)  # 20ms delay for smoother real-time viewing
                
                # If pole falls, reset and continue
                if done:
                    print(f"Episode {episode_count}: Steps = {steps}, Reward = {episode_reward:.2f}")
                    state, _ = env.reset()
                    episode_reward = 0
                    steps = 0
                    episode_count += 1
                
        except KeyboardInterrupt:
            print("\n\nStopped by user (Ctrl+C)")
    
    else:
        # Run single episode demo
        print("Running DEMO mode - single episode")
        print("Use --continuous flag for indefinite balancing\n")
        
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            action = agent.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            steps += 1
            
            # Slow down for better viewing
            time.sleep(1)  # 1000ms delay for smoother real-time viewing
            
            if done or steps >= 1000:
                break
        
        print(f"Episode completed: Steps = {steps}, Reward = {episode_reward:.2f}")
    
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='results/best_model.pt')
    parser.add_argument('--continuous', action='store_true', help='Run indefinitely (Ctrl+C to stop)')
    parser.add_argument('--no-render', action='store_true', help='Run without visualization')
    args = parser.parse_args()
    
    visualize(args.model, render=not args.no_render, continuous=args.continuous)

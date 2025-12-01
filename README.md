# TD3 Reinforcement Learning - CartPole Balancing

A GPU-accelerated implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) for training an agent to balance a CartPole (inverted pendulum). This project includes training with multiple seeds, evaluation, visualization, and learning curve plotting.

## Features

- **TD3 Algorithm**: State-of-the-art actor-critic reinforcement learning
- **GPU-Only Training**: Optimized for CUDA-enabled GPUs
- **Multi-Seed Training**: Trains with seeds [0, 1, 2] for robust evaluation
- **Evaluation Protocol**: Evaluates trained agents with seed 10
- **Real-Time Visualization**: Watch your trained agent balance the pole
- **Learning Curves**: Automatic plotting of training progress with mean ± std
- **Model Checkpointing**: Saves best performing model

## Requirements

- Python 3.8+
- CUDA-capable GPU
- NVIDIA drivers installed

## Installation

1. Clone this repository:
```bash
git clone https://github.com/vijayramsriram/cartpole.git
cd cartpole
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the TD3 agent on the CartPole task (InvertedPendulum-v4 environment):

```bash
python train.py --episodes 300
```

**Options:**
- `--episodes`: Number of training episodes (default: 300)
- `--eval-episodes`: Number of evaluation episodes (default: 10)
- `--env`: Environment name (default: InvertedPendulum-v4)

**Training Details:**
- Training seeds: 0, 1, 2
- Evaluation seed: 10
- Episodes per seed: 300 (configurable)

**Output:**
- `results/best_model.pt`: Best trained model checkpoint
- `results/learning_curve.png`: Training and evaluation performance plot

### Visualization

Watch your trained agent balance the CartPole in real-time:

**Single Episode Demo:**
```bash
python visualize.py
```

**Continuous Balancing (runs until Ctrl+C):**
```bash
python visualize.py --continuous
```

**Options:**
- `--model`: Path to model checkpoint (default: results/best_model.pt)
- `--continuous`: Run indefinitely until interrupted
- `--no-render`: Run without visual rendering

## Project Structure

```
.
├── train.py           # Main training script with TD3 implementation
├── visualize.py       # Visualization script for trained agents
├── requirements.txt   # Python dependencies
├── .gitignore        # Git ignore rules
└── results/          # Generated during training
    ├── best_model.pt
    └── learning_curve.png
```

## How It Works

### TD3 Algorithm

TD3 improves upon DDPG with three key innovations:

1. **Twin Critics**: Uses two Q-networks and takes the minimum to reduce overestimation
2. **Delayed Policy Updates**: Updates the policy less frequently than the Q-networks
3. **Target Policy Smoothing**: Adds noise to target actions for more robust learning

### Training Process

1. Agent interacts with CartPole environment collecting experiences
2. Experiences stored in replay buffer (capacity: 100,000)
3. Mini-batches sampled for training (batch size: 256)
4. Critic networks updated to minimize TD error
5. Actor network updated (delayed) to maximize Q-value
6. Target networks soft-updated with τ=0.005

### Network Architecture

- **Actor**: State (4D) → 256 → ReLU → 256 → ReLU → Action (1D, Tanh)
- **Critic (Twin)**: [State, Action] → 256 → ReLU → 256 → ReLU → Q-value

### Environment Details

**CartPole (InvertedPendulum-v4):**
- **State space**: 4-dimensional (cart position, cart velocity, pole angle, pole angular velocity)
- **Action space**: 1-dimensional continuous force applied to the cart
- **Reward**: +1 for each timestep the pole remains upright
- **Termination**: Pole angle exceeds threshold or cart moves too far

## Training Results

The training script:
- Trains 3 agents with random seeds 0, 1, and 2
- Evaluates all agents with seed 10
- Selects and saves the best performing model
- Generates learning curves showing mean ± std performance

Expected performance on CartPole (InvertedPendulum-v4):
- Training reward (final 10 episodes): ~900-1000
- Evaluation reward (seed 10): ~950-1000

### Sample Learning Curve

![Learning Curve](https://github.com/vijayramsriram/cartpole/blob/main/results/learning_curve.png)

*Figure: TD3 learning curve showing training performance (mean ± std) across seeds 0, 1, 2 and evaluation results with seed 10.*

## Hyperparameters

Key hyperparameters used in the implementation:

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Discount factor (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Policy noise | 0.2 |
| Noise clip | 0.5 |
| Policy update frequency | 2 |
| Batch size | 256 |
| Replay buffer capacity | 100,000 |
| Exploration noise | 0.1 |
| Hidden layer size | 256 |

## GPU Requirements

This implementation requires a CUDA-capable GPU. The script will raise an error if CUDA is not available.

**Verify your GPU setup:**
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

**CUDA not available:**
- Verify NVIDIA drivers: `nvidia-smi`
- Check PyTorch installation includes CUDA support
- Reinstall PyTorch with CUDA: Visit [pytorch.org](https://pytorch.org)

**Poor performance:**
- Increase training episodes (--episodes 500)
- Adjust hyperparameters in TD3Agent initialization
- Try different random seeds

**Visualization issues:**
- Ensure display is available (not headless server)
- Use `--no-render` flag for headless environments
- Check gymnasium rendering dependencies

## Academic Context

This implementation was developed as part of EEE598: Reinforcement Learning in Robotics at Arizona State University. The project demonstrates the application of TD3 algorithm to continuous control tasks with emphasis on reproducibility and clarity.

## License

MIT License

## References

- [TD3 Paper](https://arxiv.org/abs/1802.09477): "Addressing Function Approximation Error in Actor-Critic Methods"
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [DeepMind Control Suite](https://github.com/deepmind/dm_control)

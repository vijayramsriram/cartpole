# TD3 Reinforcement Learning - InvertedPendulum

A GPU-accelerated implementation of Twin Delayed Deep Deterministic Policy Gradient (TD3) for training an agent to balance an inverted pendulum. This project includes training with multiple seeds, evaluation, visualization, and learning curve plotting.

## Features

- **TD3 Algorithm**: State-of-the-art actor-critic reinforcement learning
- **GPU-Only Training**: Optimized for CUDA-enabled GPUs
- **Multi-Seed Training**: Trains with seeds [0, 1, 2] for robust evaluation
- **Real-Time Visualization**: Watch your trained agent balance the pole
- **Learning Curves**: Automatic plotting of training progress
- **Model Checkpointing**: Saves best performing model

## Requirements

- Python 3.8+
- CUDA-capable GPU
- NVIDIA drivers installed

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

Train the TD3 agent on the InvertedPendulum-v4 environment:

```bash
python train.py --episodes 300
```

**Options:**
- `--episodes`: Number of training episodes (default: 300)
- `--eval-episodes`: Number of evaluation episodes (default: 10)
- `--env`: Environment name (default: InvertedPendulum-v4)

**Output:**
- `results/best_model.pt`: Best trained model checkpoint
- `results/learning_curve.png`: Training and evaluation performance plot

### Visualization

Watch your trained agent in action:

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

1. Agent interacts with environment collecting experiences
2. Experiences stored in replay buffer (capacity: 100,000)
3. Mini-batches sampled for training (batch size: 256)
4. Critic networks updated to minimize TD error
5. Actor network updated (delayed) to maximize Q-value
6. Target networks soft-updated with τ=0.005

### Network Architecture

- **Actor**: State → 256 → ReLU → 256 → ReLU → Action (Tanh)
- **Critic (Twin)**: [State, Action] → 256 → ReLU → 256 → ReLU → Q-value

## Training Results

The training script:
- Trains 3 agents with different random seeds
- Evaluates all agents with seed 10
- Selects and saves the best performing model
- Generates learning curves showing mean ± std performance

Expected performance on InvertedPendulum-v4:
- Training reward (final 10 episodes): ~900-1000
- Evaluation reward: ~950-1000

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

## License

MIT License

## References

- [TD3 Paper](https://arxiv.org/abs/1802.09477): "Addressing Function Approximation Error in Actor-Critic Methods"
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

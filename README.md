# Adaptive Latent Distributions for Efficient Exploration in Deep Reinforcement Learning

This repository hosts the implementation for **"Adaptive Latent Distributions for Improved Exploration in Deep Reinforcement Learning"** 

Effective exploration in high-dimensional environments is a core challenge in deep reinforcement learning (RL). This work presents **Adaptive Latent Distributions for Randomized Rewards (ALD-R)**, a novel approach that improves exploration efficiency by dynamically adjusting latent reward distributions based on state coverage metrics. ALD-R enhances exploration in complex tasks with sparse rewards and high-dimensional state spaces, achieving superior results on benchmarks like Atari and IsaacGym compared to static methods.

---

## Setup Instructions

Follow these steps to install dependencies and set up the environment.

### Clone the Repository

Start by cloning this repository:

```bash
git clone https://github.com/DhruvShah09/AdaptiveLatentDistributionPPO.git
```

### Create a Conda Environment

Set up a Conda environment and activate it:

```bash
conda create -n ald-r python=3.8
conda activate ald-r
```

### Install IsaacGym

Refer to the [IsaacGymEnvs repository](https://github.com/isaac-sim/IsaacGymEnvs) for installation instructions. Ensure you can run IsaacGym examples successfully.

### Install IsaacGymEnvs

Once IsaacGym is set up, install the IsaacGymEnvs package:

```bash
cd isaacgym/IsaacGymEnvs
pip install -e .
```

### Install Python Dependencies

Install all required Python libraries:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

The repository includes scripts for running experiments on Atari and IsaacGym environments.

### Atari Benchmarks

To evaluate ALD-R on Atari games, execute the following command:

```bash
python atari/ppo_aldr.py
```
Note: you must have an existing Weights and Biases account to track the loss using WandB. Either create an account or set the appropriate flag. 

## Command-Line Flags Description

This section describes all configurable command-line flags available in the script.

### General
- `--exp-name`: (str) The name of the experiment. Defaults to the script filename.
- `--seed`: (int) Random seed for reproducibility. Default: 1.
- `--torch-deterministic`: (bool) Enforces deterministic behavior in PyTorch. Default: True.
- `--cuda`: (bool) Whether to use CUDA for GPU acceleration. Default: True.
- `--track`: (bool) Enables experiment tracking (e.g., with WandB). Default: True.
- `--wandb-project-name`: (str) Name of the WandB project for logging. Default: "random-latent-exploration".
- `--wandb-entity`: (str) WandB entity for tracking. Default: None.
- `--capture-video`: (bool) Records training videos during evaluation. Default: False.
- `--capture-video-interval`: (int) Interval (in episodes) to capture video. Default: 10.
- `--gpu-id`: (int) GPU ID to use for computations. Default: 0.

### Algorithm Configuration
- `--env-id`: (str) Environment ID for the task. Default: "Alien-v5".
- `--total-timesteps`: (int) Total number of timesteps for training. Default: 40,000,000.
- `--learning-rate`: (float) Learning rate for optimization. Default: 3e-4.
- `--num-envs`: (int) Number of environments to run in parallel. Default: 128.
- `--num-steps`: (int) Number of steps per environment per update. Default: 128.
- `--anneal-lr`: (bool) Whether to anneal the learning rate over training. Default: False.
- `--gamma`: (float) Discount factor for rewards. Default: 0.999.
- `--gae-lambda`: (float) Lambda for Generalized Advantage Estimation (GAE). Default: 0.95.
- `--num-minibatches`: (int) Number of minibatches for training updates. Default: 4.
- `--update-epochs`: (int) Number of epochs per policy update. Default: 4.
- `--norm-adv`: (bool) Whether to normalize advantages. Default: True.
- `--clip-coef`: (float) Clipping coefficient for PPO. Default: 0.1.
- `--clip-vloss`: (bool) Clips value loss in PPO. Default: True.
- `--ent-coef`: (float) Coefficient for entropy regularization. Default: 0.01.
- `--vf-coef`: (float) Coefficient for value function loss. Default: 0.5.
- `--int-vf-coef`: (float) Coefficient for intrinsic value loss. Default: 0.5.
- `--max-grad-norm`: (float) Maximum gradient norm for clipping. Default: 0.5.
- `--target-kl`: (float) Target KL divergence for stopping updates. Default: None.
- `--sticky-action`: (bool) Enables sticky actions in the environment. Default: True.
- `--normalize-ext-rewards`: (bool) Normalizes external rewards. Default: True.

### Evaluation
- `--eval-interval`: (int) Interval for evaluation (in timesteps). Default: 0.
- `--num-eval-envs`: (int) Number of environments for evaluation. Default: 32.
- `--num-eval-episodes`: (int) Number of episodes to evaluate. Default: 32.

### Random Latent Exploration (RLE)
- `--switch-steps`: (int) Interval (in steps) for switching latent rewards. Default: 500.
- `--norm-rle-features`: (bool) Normalize RLE features. Default: True.
- `--int-coef`: (float) Coefficient for intrinsic rewards. Default: 0.01.
- `--ext-coef`: (float) Coefficient for extrinsic rewards. Default: 1.0.
- `--int-gamma`: (float) Discount factor for intrinsic rewards. Default: 0.99.
- `--feature-size`: (int) Size of the latent feature vector. Default: 16.
- `--tau`: (float) Target network update rate for soft updates. Default: 0.005.
- `--save-rle`: (bool) Save RLE features for analysis. Default: False.
- `--num-iterations-feat-norm-init`: (int) Number of iterations to initialize feature normalization. Default: 1.

### Adaptive Latent Distributions (ALD-R)
- `--adapt-latent`: (bool) Enables adaptive latent distribution updates. Default: True.
- `--latent-lr`: (float) Learning rate for optimizing latent distributions. Default: 1e-3.

### Other
- `--z-layer-init`: (str) Initialization method for the latent layer. Default: "ortho_1.41:0.0".
- `--local-dir`: (str) Directory for saving results. Default: "./results".
- `--use-local-dir`: (bool) Use a local directory instead of remote storage. Default: False.

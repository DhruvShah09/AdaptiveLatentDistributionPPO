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
python atari/ppo_ald_r.py
```

### IsaacGym Benchmarks

To test ALD-R on IsaacGym tasks, use this command:

```bash
python isaacgym/ppo_ald_r.py
```

---

Feel free to reach out if you have questions or need further assistance with this codebase!
```

This markdown version is concise, rephrased, and ready for use in a project repository. Let me know if you need further adjustments!
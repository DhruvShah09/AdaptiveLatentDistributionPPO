# This code builds on the original provided code and introduces a predictor network in the AdaptiveLatentDistribution
# to ensure the coverage objective is differentiable w.r.t. the latent distribution parameters.
# The coverage metric is now computed on predicted embeddings from sampled z, rather than static environment embeddings.
import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool
import matplotlib.pyplot as plt
import seaborn as sns
import functools
from collections import defaultdict

import envpool
import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset
from copy import deepcopy

if os.environ.get("WANDB_MODE", "online") == "offline":
    from wandb_osh.hooks import TriggerWandbSyncHook
    trigger_sync = TriggerWandbSyncHook()
else:
    def dummy():
        pass
    trigger_sync = dummy

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="random-latent-exploration")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--capture-video-interval", type=int, default=10)
    parser.add_argument("--gpu-id", type=int, default=0)

    # Algorithm
    parser.add_argument("--env-id", type=str, default="Pong-v5")
    parser.add_argument("--total-timesteps", type=int, default=5000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--int-vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument("--sticky-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--normalize-ext-rewards", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

    # Evaluation
    parser.add_argument("--eval-interval", type=int, default=0)
    parser.add_argument("--num-eval-envs", type=int, default=32)
    parser.add_argument("--num-eval-episodes", type=int, default=32)

    # RLE
    parser.add_argument("--switch-steps", type=int, default=500)
    parser.add_argument("--norm-rle-features", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--int-coef", type=float, default=0.01)
    parser.add_argument("--ext-coef", type=float, default=1.0)
    parser.add_argument("--int-gamma", type=float, default=0.99)
    parser.add_argument("--feature-size", type=int, default=16)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--save-rle", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--num-iterations-feat-norm-init", type=int, default=1)

    parser.add_argument("--z-layer-init", type=str, default="ortho_1.41:0.0")

    parser.add_argument("--local-dir", type=str, default="./results")
    parser.add_argument("--use-local-dir", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # ALD-R
    parser.add_argument("--adapt-latent", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--latent-lr", type=float, default=1e-3)
    # fmt: on
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def compute_state_visitation(env_states, resolution=10):
    """
    Compute state visitation frequencies for visualization.
    Args:
        env_states (list of tuples): List of (x, y) state coordinates.
        resolution (int): The size of bins for heatmap aggregation.

    Returns:
        heatmap: A 2D array of visitation counts.
    """
    x_coords, y_coords = zip(*env_states)
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=resolution)
    return heatmap

# def plot_state_diversity(heatmap, filename="state_diversity.png"):
#     os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
#     plt.figure(figsize=(6, 6))
#     sns.heatmap(np.log1p(heatmap), cmap="viridis", cbar=True, square=True)
#     plt.title("State Visitation Count (Log Scale)")
#     plt.xlabel("State Space X")
#     plt.ylabel("State Space Y")
#     plt.savefig(filename)
#     plt.close()

# def plot_coverage_over_time(unique_states_over_time, filename="coverage_over_time.png"):
#     os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure the directory exists
#     plt.figure(figsize=(8, 6))
#     plt.plot(range(len(unique_states_over_time)), unique_states_over_time, label="Cumulative Unique States")
#     plt.xlabel("Training Steps (x1k)")
#     plt.ylabel("Unique States")
#     plt.title("Coverage Over Time")
#     plt.legend()
#     plt.grid()
#     plt.savefig(filename)
#     plt.close()



class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - (infos["terminated"] | infos["TimeLimit.truncated"])
        self.episode_lengths *= 1 - (infos["terminated"] | infos["TimeLimit.truncated"])
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (observations, rewards, dones, infos)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def sparse_layer_init(layer, sparsity=0.1, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.sparse_(layer.weight, sparsity, std=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def create_layer_init_from_spec(spec: str):
    if spec.startswith("ortho"):
        params = spec.split("_")[1].split(":")
        print(f"Create ortho init with {params}")
        return functools.partial(layer_init, std=float(params[0]), bias_const=float(params[1]))
    elif spec.startswith("sparse"):
        params = spec.split("_")[1].split(":")
        print(f"Create sparse init with {params}")
        return functools.partial(sparse_layer_init,
                                 sparsity=float(params[0]),
                                 std=float(params[1]),
                                 bias_const=float(params[2]))


class Agent(nn.Module):
    def __init__(self, envs, rle_net):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )

        self.goal_encoder = nn.Sequential(
            layer_init(nn.Linear(rle_net.feature_size, 448)),
            nn.ReLU(),
            layer_init(nn.Linear(448, 448)),
            nn.ReLU(),
        )
        self.extra_layer = nn.Sequential(layer_init(nn.Linear(448, 448), std=0.1), nn.ReLU())
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(448, envs.single_action_space.n), std=0.01),
        )
        self.critic_ext = layer_init(nn.Linear(448, 1), std=0.01)
        self.critic_int = layer_init(nn.Linear(448, 1), std=0.01)

    def get_action_and_value(self, x, reward, goal, action=None, deterministic=False):
        obs_hidden = self.network(x / 255.0)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden

        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        features = self.extra_layer(hidden)
        if action is None and not deterministic:
            action = probs.sample()
        elif action is None and deterministic:
            action = probs.probs.argmax(dim=1)
        return (action, probs.log_prob(action), probs.entropy(),
                self.critic_ext(features + hidden),
                self.critic_int(features + hidden))

    def get_value(self, x, reward, goal):
        obs_hidden = self.network(x / 255.0)
        goal_hidden = self.goal_encoder(goal)
        hidden = obs_hidden + goal_hidden
        features = self.extra_layer(hidden)
        return self.critic_ext(features + hidden), self.critic_int(features + hidden)


class Predictor(nn.Module):
    """
    A simple MLP that predicts a state embedding given a latent vector z.
    This network ensures a differentiable mapping from z -> predicted embedding.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.model(z)


class AdaptiveLatentDistribution(nn.Module):
    """
    A parametric distribution p_θ(z) for latent vectors z, implemented as a Gaussian with
    learnable mean and diagonal covariance. It also maintains a predictor network that,
    given z, predicts the state embeddings that would be visited. This allows a differentiable
    coverage metric and thus lets us adapt the latent distribution parameters to improve coverage.

    Key functionalities:
    - Maintains a replay buffer of (z, embedding) pairs from environment rollouts.
    - Trains a predictor network to map z -> predicted embedding.
    - Computes a coverage metric on predicted embeddings sampled from p_θ(z), ensuring
      that coverage objective is differentiable w.r.t. mean and log_std.
    - Supports methods to prune old data and create datasets for predictor training.

    Attributes:
        feature_size (int): Dimension of latent vectors and embeddings.
        device (torch.device): The device on which computations run.
        mean (nn.Parameter): Mean of the Gaussian distribution for z.
        log_std (nn.Parameter): Log standard deviation of the Gaussian distribution for z.
        predictor (Predictor): MLP that maps z -> predicted embeddings.
        predictor_optimizer (optim.Optimizer): Optimizer for the predictor network.
        z_storage (List[torch.Tensor]): Buffer storing z samples collected over time.
        emb_storage (List[torch.Tensor]): Buffer storing embeddings corresponding to z samples.
        max_storage_size (int): Maximum size of the replay buffer.
    """
    def __init__(self, feature_size: int, device: torch.device, max_storage_size: int = 100000):
        super().__init__()
        self.feature_size = feature_size
        self.device = device
        self.max_storage_size = max_storage_size

        # Latent distribution parameters
        self.mean = nn.Parameter(torch.zeros(feature_size, device=device))
        self.log_std = nn.Parameter(torch.zeros(feature_size, device=device))

        # Predictor for z -> embedding
        self.predictor = Predictor(feature_size, feature_size).to(device)
        self.predictor_optimizer = optim.Adam(self.predictor.parameters(), lr=1e-2)

        # Replay buffer for (z, embedding) pairs
        self.z_storage = []
        self.emb_storage = []

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Sample 'num_samples' latent vectors from p_θ(z).

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: (num_samples, feature_size) of sampled z.
        """
        std = torch.exp(self.log_std)
        dist = Normal(self.mean, std)
        return dist.sample((num_samples,))

    def forward(self, num_samples: int = 1) -> torch.Tensor:
        """
        Quickly sample z for forward calls if needed.
        """
        return self.sample(num_samples)

    def store_embeddings(self, z_samples: torch.Tensor, state_embeddings: torch.Tensor):
        """
        Store (z, embedding) pairs in the replay buffer. Both z_samples and state_embeddings
        are expected to be on CPU or will be moved to CPU for storage.

        Args:
            z_samples (torch.Tensor): (N, feature_size) z vectors used during rollouts.
            state_embeddings (torch.Tensor): (N, feature_size) actual embeddings from environment states.
        """
        # Move to CPU for storage
        z_cpu = z_samples.detach().cpu()
        emb_cpu = state_embeddings.detach().cpu()
        self.z_storage.append(z_cpu)
        self.emb_storage.append(emb_cpu)

    def prune_old_data(self):
        """
        Keep the replay buffer size manageable by pruning old data if it exceeds max_storage_size.
        We keep the most recent data.
        """
        if len(self.z_storage) == 0:
            return

        z_all = torch.cat(self.z_storage, dim=0)
        emb_all = torch.cat(self.emb_storage, dim=0)

        total = z_all.size(0)
        if total > self.max_storage_size:
            excess = total - self.max_storage_size
            z_all = z_all[excess:]
            emb_all = emb_all[excess:]

        self.z_storage = [z_all]
        self.emb_storage = [emb_all]

    def create_predictor_dataset(self) -> TensorDataset:
        """
        Create a PyTorch Dataset from all stored (z, embedding) pairs.

        Returns:
            TensorDataset: dataset containing (z, embedding) pairs.
        """
        if len(self.z_storage) == 0:
            z_all = torch.empty((0, self.feature_size))
            emb_all = torch.empty((0, self.feature_size))
        else:
            z_all = torch.cat(self.z_storage, dim=0)
            emb_all = torch.cat(self.emb_storage, dim=0)

        return TensorDataset(z_all, emb_all)

    def coverage_objective(self) -> torch.Tensor:
        """
        Compute a coverage metric based on predicted embeddings for sampled z.

        Steps:
        1. Sample multiple z from the current distribution.
        2. Predict embeddings for these z using the predictor.
        3. Compute the average pairwise distance between these predicted embeddings as coverage.

        Returns:
            torch.Tensor: scalar coverage metric.
        """
        num_samples = 64  # Arbitrary chosen number of samples for coverage estimation
        z_samples = self.sample(num_samples)  # (64, feature_size)
        predicted_embeddings = self.predictor(z_samples)  # (64, feature_size)

        distances = torch.cdist(predicted_embeddings, predicted_embeddings, p=2)
        N = distances.size(0)
        diag_mask = torch.ones((N, N), device=self.device, dtype=torch.bool)
        diag_mask.fill_diagonal_(False)
        off_diag_distances = distances[diag_mask]

        coverage_value = off_diag_distances.mean()
        return coverage_value


class RLEModel(nn.Module):
    def __init__(self, input_size, feature_size, output_size, num_actions, num_envs, z_layer_init, device, args):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.feature_size = feature_size
        self.num_envs = num_envs
        self.switch_steps = args.switch_steps
        self.args = args

        self.rle_net = nn.Sequential(
            layer_init(nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 448)),
            nn.ReLU(),
        )
        self.last_layer = z_layer_init(nn.Linear(448, self.feature_size))

        if self.args.adapt_latent:
            self.latent_dist = AdaptiveLatentDistribution(self.feature_size, self.device)
        else:
            self.latent_dist = None

        self.goals = self._sample_goals()

        self.num_steps_left = torch.randint(1, self.switch_steps, (num_envs,)).to(device)
        self.switch_goals_mask = torch.zeros(num_envs).to(device)

        self.rle_rms = RunningMeanStd(shape=(1, self.feature_size))
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def _sample_goals(self, num_envs=None):
        if num_envs is None:
            num_envs = self.num_envs
        if self.args.adapt_latent:
            goals = self.latent_dist.sample(num_envs)
            goals = goals / torch.norm(goals, dim=1, keepdim=True)
        else:
            goals = torch.randn((num_envs, self.feature_size), device=self.device).float()
            goals = goals / torch.norm(goals, dim=1, keepdim=True)
        return goals

    def step(self, next_done: torch.Tensor, next_ep_done: torch.Tensor):
        self.switch_goals_mask = torch.zeros(self.num_envs).to(self.device)
        self.switch_goals_mask[next_done.bool()] = 1.0
        self.num_steps_left -= 1
        self.switch_goals_mask[self.num_steps_left == 0] = 1.0

        new_goals = self._sample_goals()
        self.goals = self.goals * (1 - self.switch_goals_mask.unsqueeze(1)) + new_goals * self.switch_goals_mask.unsqueeze(1)
        self.num_steps_left[self.switch_goals_mask.bool()] = self.switch_steps

        return self.switch_goals_mask

    def compute_rle_feat(self, obs, goals=None):
        if goals is None:
            goals = self.goals
        with torch.no_grad():
            raw_rle_feat = self.last_layer(self.rle_net(obs))
            rle_feat = (raw_rle_feat - self.rle_feat_mean) / (self.rle_feat_std + 1e-5)
            reward = (rle_feat * goals).sum(axis=1) / torch.norm(rle_feat, dim=1)
        return reward, raw_rle_feat, rle_feat

    def update_rms(self, b_rle_feats):
        self.rle_rms.update(b_rle_feats)
        self.rle_feat_mean = torch.tensor(self.rle_rms.mean, device=self.device).float()
        self.rle_feat_std = torch.sqrt(torch.tensor(self.rle_rms.var, device=self.device)).float()

    def compute_reward(self, obs, next_obs, goals=None):
        return self.compute_rle_feat(next_obs, goals=goals)


class RewardForwardFilter:
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews, not_done=None):
        if not_done is None:
            if self.rewems is None:
                self.rewems = rews
            else:
                self.rewems = self.rewems * self.gamma + rews
            return self.rewems
        else:
            if self.rewems is None:
                self.rewems = rews
            else:
                mask = np.where(not_done == 1.0)
                self.rewems[mask] = self.rewems[mask] * self.gamma + rews[mask]
            return deepcopy(self.rewems)


class VideoRecorder(object):
    def __init__(self,
                 max_buffer_size: int = 27000 * 5,
                 local_dir: str = "./results",
                 use_wandb: bool = False,
                 use_local_dir: bool = False) -> None:
        self.use_wandb = use_wandb
        self.local_dir = local_dir
        self.use_local_dir = use_local_dir
        self.max_buffer_size = max_buffer_size
        self.frame_buffer = deque(maxlen=max_buffer_size)
        self.rewards = deque(maxlen=max_buffer_size)
        self.int_rewards = deque(maxlen=max_buffer_size)
        self.episode_count = 0
        self.fig = plt.figure()

    def record(self, frames: np.ndarray, rewards: float, int_reward_info: dict, global_step: int):
        self.frame_buffer.append(np.expand_dims(frames, axis=0).astype(np.uint8))
        self.rewards.append(rewards)
        self.int_rewards.append(int_reward_info["int_rewards"])

    def reset(self):
        self.frame_buffer.clear()
        self.rewards.clear()
        self.int_rewards.clear()
        self.episode_count += 1

    def flush(self, global_step: int, caption: str = ""):
        if len(self.frame_buffer) <= 0:
            return
        if len(caption) <= 0:
            caption = f"episode-{self.episode_count}-score-{np.stack(self.rewards).sum()}"

        video_array = np.concatenate(self.frame_buffer, axis=0)
        video_array = video_array[:, None, ...]
        save_path = os.path.join(self.local_dir, str(self.episode_count), str(caption))
        print(f"Log frames and rewards at {save_path}")
        if self.use_local_dir:
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "frames.npy"), video_array)
            np.save(os.path.join(save_path, "rewards.npy"), np.stack(self.rewards))
            np.save(os.path.join(save_path, "int_rewards.npy"), np.stack(self.int_rewards))

        if self.use_wandb:
            # wandb must have been imported if track is True
            wandb.log({"media/video": wandb.Video(video_array, fps=30, caption=str(caption))}, step=global_step)
            task_lineplot = sns.lineplot(np.stack(self.rewards))
            log_data = wandb.Image(self.fig)
            wandb.log({"media/task_rewards": log_data}, step=global_step)
            plt.clf()

            int_reward_lineplot = sns.lineplot(np.stack(self.int_rewards))
            log_data = wandb.Image(self.fig)
            wandb.log({"media/int_reward": log_data}, step=global_step)
            plt.clf()

        self.reset()



class VideoRecordScoreCondition:
    score_thresholds = [
        0,100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,
        1600,1700,1800,2500,3000,5000,6000,7000,8000,9000,10000,20000,30000,
        40000,50000,60000,70000,80000,90000,100000, np.inf,
    ]

    def __init__(self) -> None:
        self.has_recorded = pd.DataFrame({"value": [False] * (len(self.score_thresholds) - 1)},
                                         index=pd.IntervalIndex.from_breaks(self.score_thresholds, closed='left'))
        print("Record score intervals: ", self.score_thresholds)

    def __call__(self, score: float, global_step: int):
        if not self.has_recorded.iloc[self.has_recorded.index.get_loc(score)]["value"]:
            print(f"Record the first video with score {score}")
            self.has_recorded.iloc[self.has_recorded.index.get_loc(score)] = True
            return True
        return False


class VideoStepConditioner:
    def __init__(self, global_step_interval: int) -> None:
        self.global_step_interval = global_step_interval
        self.last_global_step = 0

    def __call__(self, score: float, global_step: int):
        if global_step - self.last_global_step >= self.global_step_interval:
            self.last_global_step = global_step
            return True
        return False


if __name__ == "__main__":
    fig = plt.figure()
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, config=vars(args), name=run_name, save_code=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() and args.cuda else "cpu")
    print("Device: ", device)

    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        max_episode_steps=int(108000 / 4),
        seed=args.seed,
        repeat_action_probability=0.25,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    rle_output_size = args.feature_size
    num_actions = envs.single_action_space.n
    rle_network = RLEModel(envs.single_observation_space.shape, args.feature_size, rle_output_size, num_actions,
                           args.num_envs, z_layer_init=create_layer_init_from_spec(args.z_layer_init),
                           device=device, args=args).to(device)
    rle_feature_size = rle_network.feature_size

    agent = Agent(envs, rle_network).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.adapt_latent and rle_network.latent_dist is not None:
        latent_optimizer = optim.Adam([rle_network.latent_dist.mean, rle_network.latent_dist.log_std],
                                      lr=args.latent_lr)

    int_reward_rms = RunningMeanStd()
    int_discounted_reward = RewardForwardFilter(args.int_gamma)
    ext_reward_rms = RunningMeanStd()
    ext_discounted_reward = RewardForwardFilter(args.gamma)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    next_obss = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    goals = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rle_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)

    prev_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rle_dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    raw_rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)
    rle_feats = torch.zeros((args.num_steps, args.num_envs, rle_feature_size)).to(device)

    avg_returns = deque(maxlen=128)
    avg_ep_lens = deque(maxlen=128)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    next_raw_rle_feat = []
    if args.norm_rle_features:
        print("Start to initialize rle features normalization parameter.....")
        init_start = time.time()
        for step in range(args.num_steps * args.num_iterations_feat_norm_init):
            acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
            s, r, d, _ = envs.step(acs)
            rle_reward, raw_rle_feat, rle_feat = rle_network.compute_rle_feat(torch.Tensor(s).to(device).float())
            next_raw_rle_feat += raw_rle_feat.detach().cpu().numpy().tolist()

            if len(next_raw_rle_feat) % (args.num_steps * args.num_envs) == 0:
                next_raw_rle_feat = np.stack(next_raw_rle_feat)
                rle_network.update_rms(next_raw_rle_feat)
                next_raw_rle_feat = []
        print(f"End of initializing... finished in {time.time() - init_start}")

    video_recorder = VideoRecorder(local_dir=args.local_dir,
                               use_wandb=args.track & args.capture_video,
                               use_local_dir=args.use_local_dir)

    video_record_conditioner = VideoStepConditioner(global_step_interval=int(args.capture_video_interval))
    is_early_stop = False


    visited_states = []
    unique_states = set()
    unique_states_over_time = []
    visualization_interval = 50

    for update in range(1, num_updates + 1):
        prev_rewards[0] = rle_rewards[-1] * args.int_coef if update > 1 else 0
        it_start_time = time.time()

        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            if args.adapt_latent:
                latent_optimizer.param_groups[0]["lr"] = frac * args.latent_lr

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            rle_dones[step] = rle_network.switch_goals_mask
            goals[step] = rle_network.goals

            with torch.no_grad():
                action, logprob, _, value_ext, value_int = agent.get_action_and_value(
                    next_obs, prev_rewards[step], goals[step], deterministic=False
                )
                ext_values[step], int_values[step] = (value_ext.flatten(), value_int.flatten())
            actions[step] = action
            logprobs[step] = logprob

            next_obs_, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs_).to(device), torch.Tensor(done).to(device)
            next_obss[step] = next_obs
            rle_obs = obs[step].float()
            rle_next_obs = next_obs.clone().float()
            rle_reward, raw_next_rle_feat, next_rle_feat = rle_network.compute_reward(rle_obs, rle_next_obs)
            rle_rewards[step] = rle_reward.data
            raw_rle_feats[step] = raw_next_rle_feat
            rle_feats[step] = next_rle_feat

            int_reward_info = {"int_rewards": rle_rewards[step, 0].cpu().numpy()}

            state_x = next_obs_[:, 0].mean()
            state_y = next_obs_[:, 1].mean()

            visited_states.append((state_x, state_y))
            unique_states.add((state_x, state_y))
            unique_states_over_time.append(len(unique_states))


            if args.capture_video:
                video_recorder.record(obs[step][0, 3, ...].cpu().numpy().copy(), info["reward"][0], int_reward_info, global_step=global_step)

            if step < args.num_steps - 1:
                prev_rewards[step + 1] = rle_rewards[step] * args.int_coef

            for idx, d in enumerate(done):
                if info["terminated"][idx] or info["TimeLimit.truncated"][idx]:
                    avg_returns.append(info["r"][idx])
                    avg_ep_lens.append(info["l"][idx])
                    if args.track:
                        wandb.log({"charts/episode_return": info["r"][idx]}, step=global_step)

                    if args.capture_video and idx == 0:
                        if video_record_conditioner(info["r"][idx], global_step):
                            video_recorder.flush(global_step=global_step, caption=f"{info['r'][idx]}")
                            print(f"Logged a video with len={info['l'][idx]} and return={info['r'][idx]}")
                        else:
                            video_recorder.reset()
                            print(f"Env idx={idx}: len={info['l'][idx]} and return={info['r'][idx]} is reset.")
                        trigger_sync()

            next_ep_done = info["terminated"] | info["TimeLimit.truncated"]
            rle_network.step(next_done, next_ep_done)


        
        print("update:", update)
        print("visualization_interval", visualization_interval)

        if update % visualization_interval == 0:
            # Generate heatmap of state diversity
            heatmap = compute_state_visitation(visited_states, resolution=50)  
            state_diversity_save_path = f"./results/aldr/state_diversity_{update}.npy"
            print(f"Saving state diversity data to: {state_diversity_save_path}")
            np.save(state_diversity_save_path, heatmap)
        
            # Generate coverage plot
            coverage_data_save_path = f"./results/aldr/coverage_over_time_{update}.npy"
            print(f"Saving coverage over time data to: {coverage_data_save_path}")
            np.save(coverage_data_save_path, np.array(unique_states_over_time))


        # Normalize rewards
        not_dones = (1.0 - dones).cpu().data.numpy()
        rewards_cpu = rewards.cpu().data.numpy()
        rle_rewards_cpu = rle_rewards.cpu().data.numpy()

        if args.normalize_ext_rewards:
            ext_reward_per_env = np.array([ext_discounted_reward.update(rewards_cpu[i], not_dones[i]) for i in range(args.num_steps)])
            ext_reward_rms.update(ext_reward_per_env.flatten())
            rewards /= np.sqrt(ext_reward_rms.var)

        rle_not_dones = (1.0 - rle_dones).cpu().data.numpy()
        rle_reward_per_env = np.array([int_discounted_reward.update(rle_rewards_cpu[i], rle_not_dones[i]) for i in range(args.num_steps)])
        int_reward_rms.update(rle_reward_per_env.flatten())
        rle_rewards /= np.sqrt(int_reward_rms.var)

        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs, prev_rewards[step], goals[step])
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(rle_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0 - rle_network.switch_goals_mask
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0 - rle_dones[t + 1]
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = rle_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam)
                int_advantages[t] = int_lastgaelam = (int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam)
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_next_obs = next_obss.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_goals = goals.reshape((-1, rle_feature_size))
        b_dones = dones.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)
        b_int_values = int_values.reshape(-1)
        b_raw_rle_feats = raw_rle_feats.reshape((-1, rle_feature_size))
        b_prev_rewards = prev_rewards.reshape(-1)

        if args.norm_rle_features:
            rle_network.update_rms(b_raw_rle_feats.cpu().numpy())

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_prev_rewards[mb_inds], b_goals[mb_inds], b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef, args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss * args.vf_coef + int_v_loss * args.int_vf_coef
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                for param, target_param in zip(agent.network.parameters(), rle_network.rle_net.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # After finishing PPO updates and before evaluation:
        if args.adapt_latent and rle_network.latent_dist is not None:
            # We have (b_goals, b_raw_rle_feats) from this batch.
            # Each row in b_goals corresponds to a sampled z for that timestep,
            # and b_raw_rle_feats is the actual environmental embedding.
        
            # 1. Store these pairs in the latent_dist replay buffer
            rle_network.latent_dist.store_embeddings(b_goals, b_raw_rle_feats)
        
            # 2. Update the predictor network multiple times to ensure it accurately maps z -> embeddings
            predictor_train_steps = 10  # a reasonable number of steps for predictor training
            predictor_batch_size = 256 if args.batch_size > 256 else args.batch_size
            predictor_dataset = rle_network.latent_dist.create_predictor_dataset()
            predictor_loader = torch.utils.data.DataLoader(predictor_dataset, batch_size=predictor_batch_size, shuffle=True)
        
            initial_pred_loss = 0.0
            with torch.no_grad():
                total_items = 0
                for z_batch, emb_batch in predictor_loader:
                    z_batch, emb_batch = z_batch.to(device), emb_batch.to(device)
                    pred = rle_network.latent_dist.predictor(z_batch)
                    batch_loss = ((pred - emb_batch)**2).sum().item()
                    initial_pred_loss += batch_loss
                    total_items += z_batch.size(0)
                if total_items > 0:
                    initial_pred_loss /= total_items
        
            # Train predictor
            for _ in range(predictor_train_steps):
                for z_batch, emb_batch in predictor_loader:
                    z_batch, emb_batch = z_batch.to(device), emb_batch.to(device)
                    pred = rle_network.latent_dist.predictor(z_batch)
                    pred_loss = ((pred - emb_batch)**2).mean()
                    rle_network.latent_dist.predictor_optimizer.zero_grad()
                    pred_loss.backward()
                    rle_network.latent_dist.predictor_optimizer.step()
        
            final_pred_loss = 0.0
            with torch.no_grad():
                total_items = 0
                final_loss_accum = 0.0
                for z_batch, emb_batch in predictor_loader:
                    z_batch, emb_batch = z_batch.to(device), emb_batch.to(device)
                    pred = rle_network.latent_dist.predictor(z_batch)
                    batch_loss = ((pred - emb_batch)**2).sum().item()
                    final_loss_accum += batch_loss
                    total_items += z_batch.size(0)
                if total_items > 0:
                    final_pred_loss = final_loss_accum / total_items
        
            # Prune old data in replay buffer if needed
            rle_network.latent_dist.prune_old_data()
        
            # 3. Now compute the coverage objective using the predictor:
            latent_optimizer.zero_grad()
            coverage_obj = rle_network.latent_dist.coverage_objective()
            (-coverage_obj).backward()
            latent_optimizer.step()
        
            # Log predictor training and coverage metrics
            if args.track:
                wandb.log({
                    "predictor/initial_mse_loss": initial_pred_loss,
                    "predictor/final_mse_loss": final_pred_loss,
                    "coverage/coverage_objective": coverage_obj.item(),
                }, step=global_step)
        
        # EVALUATION and LOGGING
        if args.eval_interval != 0 and update % args.eval_interval == 0:
            print(f"Evaluating at step {update}...")
            eval_start_time = time.time()
            eval_scores = []
            eval_ep_lens = []
            eval_envs = envpool.make(
                args.env_id,
                env_type="gym",
                num_envs=args.num_eval_envs,
                episodic_life=True,
                reward_clip=True,
                max_episode_steps=int(108000 / 4),
                seed=args.seed,
                repeat_action_probability=0.25,
            )
            eval_envs.num_envs = args.num_eval_envs
            eval_envs.single_action_space = eval_envs.action_space
            eval_envs.single_observation_space = eval_envs.observation_space
            eval_envs = RecordEpisodeStatistics(eval_envs)
        
            eval_obs = torch.Tensor(eval_envs.reset()).to(device)
            eval_done = torch.zeros(args.num_eval_envs).to(device)
            eval_goal = rle_network._sample_goals(args.num_eval_envs).to(device)
        
            num_steps_left = args.switch_steps * torch.ones(args.num_eval_envs).to(device)
            switch_goals_mask = torch.zeros(args.num_eval_envs).to(device)
        
            eval_prev_reward = torch.zeros((args.num_eval_envs,)).to(device)
        
            while len(eval_scores) < args.num_eval_episodes:
                with torch.no_grad():
                    eval_action, _, _, _, _ = agent.get_action_and_value(
                        eval_obs, eval_prev_reward, eval_goal, deterministic=True
                    )
                eval_obs_, eval_reward_, eval_done_, eval_info_ = eval_envs.step(eval_action.cpu().numpy())
                eval_reward_ = torch.tensor(eval_reward_).to(device).view(-1)
                eval_obs, eval_done = torch.Tensor(eval_obs_).to(device), torch.Tensor(eval_done_).to(device)
        
                for idx, d in enumerate(eval_done_):
                    if eval_info_["terminated"][idx] or eval_info_["TimeLimit.truncated"][idx]:
                        eval_scores.append(eval_info_["r"][idx])
                        eval_ep_lens.append(eval_info_["elapsed_step"][idx])
        
                switch_goals_mask = torch.zeros(args.num_eval_envs).to(device)
                num_steps_left -= 1
                switch_goals_mask[num_steps_left == 0] = 1.0
                switch_goals_mask[eval_done.bool()] = 1.0
                new_goals = rle_network._sample_goals(args.num_eval_envs).to(device)
                eval_goal = eval_goal * (1 - switch_goals_mask.unsqueeze(1)) + new_goals * switch_goals_mask.unsqueeze(1)
                num_steps_left[switch_goals_mask.bool()] = args.switch_steps
        
            eval_envs.close()
            eval_end_time = time.time()
        
            print(f"Evaluation finished in {eval_end_time - eval_start_time} seconds")
            print(f"Step {update}: game score: {np.mean(eval_scores)}")
        
            eval_data = {}
            eval_data["eval/score"] = np.mean(eval_scores)
            eval_data["eval/min_score"] = np.min(eval_scores)
            eval_data["eval/max_score"] = np.max(eval_scores)
            eval_data["eval/ep_len"] = np.mean(eval_ep_lens)
            eval_data["eval/min_ep_len"] = np.min(eval_ep_lens)
            eval_data["eval/max_ep_len"] = np.max(eval_ep_lens)
            eval_data["eval/num_episodes"] = len(eval_scores)
            eval_data["eval/time"] = eval_end_time - eval_start_time
        
            if args.track:
                wandb.log(eval_data, step=global_step)
                trigger_sync()
        
        it_end_time = time.time()
        data = {}
        data["charts/iterations"] = update
        data["charts/learning_rate"] = optimizer.param_groups[0]["lr"]
        data["charts/time_per_it"] = it_end_time - it_start_time
        data["charts/SPS"] = int(global_step / (time.time() - start_time))
        data["charts/traj_len"] = np.mean(avg_ep_lens)
        data["charts/max_traj_len"] = np.max(avg_ep_lens, initial=0)
        data["charts/min_traj_len"] = np.min(avg_ep_lens, initial=0)
        data["charts/game_score"] = np.mean(avg_returns)
        data["charts/max_game_score"] = np.max(avg_returns, initial=0)
        data["charts/min_game_score"] = np.min(avg_returns, initial=0)
        
        data["losses/policy_loss"] = pg_loss.item()
        data["losses/ext_value_loss"] = ext_v_loss.item()
        data["losses/int_value_loss"] = int_v_loss.item()
        data["losses/entropy"] = entropy_loss.item()
        data["losses/approx_kl"] = approx_kl.item()
        data["losses/clipfrac"] = np.mean(clipfracs)
        data["losses/all_loss"] = loss.item()
        
        data["rewards/rewards_mean"] = rewards.mean().item()
        data["rewards/intrinsic_rewards_mean"] = rle_rewards.mean().item()
        data["rewards/num_envs_with_pos_rews"] = torch.sum(rewards.sum(dim=0) > 0).item()
        
        data["returns/advantages"] = b_advantages.mean().item()
        data["returns/ext_advantages"] = b_ext_advantages.mean().item()
        data["returns/int_advantages"] = b_int_advantages.mean().item()
        data["returns/ret_ext"] = b_ext_returns.mean().item()
        data["returns/ret_int"] = b_int_returns.mean().item()
        data["returns/values_ext"] = b_ext_values.mean().item()
        data["returns/values_int"] = b_int_values.mean().item()
        
        if args.adapt_latent and rle_network.latent_dist is not None:
            std = torch.exp(rle_network.latent_dist.log_std)
            data["latent/mean_norm"] = torch.norm(rle_network.latent_dist.mean).item()
            data["latent/std_mean"] = std.mean().item()
        
        if update % 100 == 0 and args.track:
            data["charts/returns_hist"] = wandb.Histogram(avg_returns)
        
        if args.track:
            wandb.log(data, step=global_step)
            trigger_sync()


    envs.close()
    if args.save_rle:
        if not os.path.exists("saved_rle_networks"):
            os.makedirs("saved_rle_networks")
        torch.save(rle_network.state_dict(), f"saved_rle_networks/{run_name}.pt")

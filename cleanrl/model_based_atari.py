# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import ReplayBuffer, ReplayBufferSamples

# from utils.logging import print_header, print_section, print_success, print_metrics, print_eval_summary, print_metrics


@dataclass
class AgentConfig:
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""


@dataclass
class TrainConfig:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    log_frequency: int = 100
    """the frequency of logging metrics"""

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway

    agent: AgentConfig = AgentConfig()

    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    learning_starts: int = 2e4
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network network optimizer"""
    update_frequency: int = 4
    """the frequency of training updates"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_q = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        x = F.relu(self.fc1(x))
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.Flatten(),
        )

        with torch.inference_mode():
            output_dim = self.conv(torch.zeros(1, *obs_shape)).shape[1]

        self.fc1 = layer_init(nn.Linear(output_dim, 512))
        self.fc_logits = layer_init(nn.Linear(512, envs.single_action_space.n))

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.relu(self.fc1(x))
        logits = self.fc_logits(x)

        return logits

    def get_action(self, x):
        logits = self(x / 255.0)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


class Agent:
    def __init__(self, cfg: TrainConfig, envs: gym.vector.SyncVectorEnv, device: str):
        self.cfg = cfg

        self.actor = Actor(envs).to(device)
        self.qf1 = SoftQNetwork(envs).to(device)
        self.qf2 = SoftQNetwork(envs).to(device)
        self.qf1_target = SoftQNetwork(envs).to(device)
        self.qf2_target = SoftQNetwork(envs).to(device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.q_optimizer = optim.Adam(list(self.qf1.parameters()) + list(self.qf2.parameters()), lr=cfg.q_lr, eps=1e-4)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=cfg.policy_lr, eps=1e-4)

        # Automatic entropy tuning
        if cfg.autotune:
            self.target_entropy = -cfg.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.Adam([self.log_alpha], lr=cfg.q_lr, eps=1e-4)
        else:
            self.alpha = cfg.alpha

    def update_step(self, batch: ReplayBufferSamples, global_step: int) -> dict[str, Any]:

        # CRITIC training
        with torch.no_grad():
            _, next_state_log_pi, next_state_action_probs = self.get_action(batch.next_observations)
            qf1_next_target = self.qf1_target(batch.next_observations)
            qf2_next_target = self.qf2_target(batch.next_observations)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_qf_next_target = next_state_action_probs * (
                torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            )
            # adapt Q-target for discrete Q-function
            min_qf_next_target = min_qf_next_target.sum(dim=1)
            next_q_value = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.cfg.agent.gamma * (min_qf_next_target)

        # use Q-values only for the taken actions
        qf1_values = self.qf1(batch.observations)
        qf2_values = self.qf2(batch.observations)
        qf1_a_values = qf1_values.gather(1, batch.actions.long()).view(-1)
        qf2_a_values = qf2_values.gather(1, batch.actions.long()).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()

        # ACTOR training
        _, log_pi, action_probs = self.get_action(batch.observations)
        with torch.no_grad():
            qf1_values = self.qf1(batch.observations)
            qf2_values = self.qf2(batch.observations)
            min_qf_values = torch.min(qf1_values, qf2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (action_probs * ((self.alpha * log_pi) - min_qf_values)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.cfg.autotune:
            # reuse action probabilities for temperature loss
            alpha_loss = (action_probs.detach() * (-self.log_alpha.exp() * (log_pi + self.target_entropy).detach())).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # update the target networks
        if global_step % self.cfg.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.cfg.agent.tau * param.data + (1 - self.cfg.agent.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.cfg.agent.tau * param.data + (1 - self.cfg.agent.tau) * target_param.data)

        info = {
            "losses/qf1_values": qf1_a_values.mean().detach(),
            "losses/qf2_values": qf2_a_values.mean().detach(),
            "losses/qf1_loss": qf1_loss.detach(),
            "losses/qf2_loss": qf2_loss.detach(),
            "losses/qf_loss": qf_loss.detach() / 2.0,
            "losses/actor_loss": actor_loss.detach(),
            "losses/alpha": self.alpha,
        }
        if self.cfg.autotune:
            info.update({"losses/alpha_loss": alpha_loss.detach()})

        return info

    def get_action(self, x):
        return self.actor.get_action(x)


# if __name__ == "__main__":
def main(cfg):
    """Main training function."""
    print_header("🚀 JAX-RL Training")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # Print configuration
    print_section("Configuration")
    print(f"  Environment: {cfg.env_id}")
    print(f"  Agent: SAC")
    print(f"  Total steps: {cfg.total_timesteps}")
    # print(f"  Eval frequency: {cfg.eval_frequency}")
    print(f"  Log frequency: {cfg.log_frequency}")
    # print(f"  Save frequency: {cfg.save_frequency}")
    print(f"  Seed: {cfg.seed}")
    print(f"  Record video: {cfg.capture_video}")
    print(f"  Device: {device}")

    run_name = f"{cfg.env_id}__{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    if cfg.track:
        import wandb

        # Initialize Wandb
        print_section("Initializing Wandb")

        wandb_run = wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=vars(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        print_success(f"Wandb run: {wandb_run.name}")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(cfg).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    # Create environment
    print_section("Creating Environment")
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id, cfg.seed, 0, cfg.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Create agent
    print_section("Creating Agent")
    agent = Agent(cfg, envs=envs, device=device)

    rb = ReplayBuffer(
        cfg.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=cfg.seed)
    for global_step in range(cfg.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < cfg.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = agent.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # print_metrics(global_step, info, eval_mode=False)
                # print_eval_summary(global_step, length=info["episode"]["l"].item(), reward=info["episode"]["r"].item(), success=None)
                # metrics = {
                #     "charts/episodic_return": info["episode"]["r"],
                #     "charts/episodic_length": info["episode"]["l"],
                # }
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > cfg.learning_starts:
            if global_step % cfg.update_frequency == 0:
                batch = rb.sample(cfg.batch_size)
                metrics.update(agent.update_step(batch, global_step=global_step))

            if global_step % cfg.log_frequency == 0:
                info.update({"charts/SPS": int(global_step / (time.time() - start_time))})
                for key, value in info.items():
                    writer.add_scalar(key, value.item() if isinstance(value, torch.Tensor) else value, global_step)

                # print_metrics(global_step, info, eval_mode=False)

    # Clean up
    envs.close()
    writer.close()
    # print_success("Training completed successfully!")


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
import os
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from einops import einsum, rearrange
from info_nce import InfoNCE, info_nce
from tensordict import TensorDict
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# import cleanrl.utils.helper as h
from cleanrl.utils.layers import mlp
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from cleanrl_utils.buffers import (
    ReplayBuffer,
    ReplayBufferLatentTrajSamples,
    ReplayBufferTrajSamples,
)

# from utils.logging import (
#     print_eval_summary,
#     print_header,
#     print_metrics,
#     print_section,
#     print_success,
# )


@dataclass
class AgentConfig:
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """target smoothing coefficient (default: 1)"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """the learning rate of the Q network optimizer"""
    batch_size: int = 64
    """the batch size of sample from the reply memory"""
    target_network_frequency: int = 8000
    """the frequency of updates for the target networks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    target_entropy_scale: float = 0.89
    """coefficient for scaling the autotune entropy target"""

    mlp_dims: List[int] = field(default_factory=lambda: [512, 512])
    """MLP dims for actor/critic/reward"""

    model_lr: float = 3e-4
    """the learning rate of the world model network optimizer"""
    latent_dim: int = 100
    """Size of latent space"""
    horizon: int = 5
    """Horizon used for representation learning"""
    consistency_coef: float = 1.0
    """Weight the temporal consistency loss by consistency_coef"""
    reward_coef: float = 1.0
    """Weight the reward loss by reward_coef"""
    rho: float = 0.9
    """Discount factor for representation learning"""
    consistency_loss: str = "mse"  # "cross-entropy", "mse", "cosine"
    """Which loss function to use for consistency loss?"""

    use_projection: bool = False
    """If true, calculate the loss in a projected space"""
    use_horizon_as_negatives: bool = False

    use_delta: bool = False
    """Predict change in latent or next latent? i.e. next_z = z + f(z, a) else next_z = f(z, a)"""

    compile: bool = False
    """If True try to compile all NNs"""


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
    learning_starts: int = 2e4
    """timestep to start learning"""
    update_frequency: int = 4
    """the frequency of training updates"""

    # Algorithm specific arguments
    env_id: str = "BeamRiderNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 5000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""  # smaller than in original paper but evaluation is done only for 100k steps anyway

    agent: AgentConfig = AgentConfig()


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


class Encoder(nn.Module):
    def __init__(self, envs, cfg: AgentConfig):
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

        self.fc1 = layer_init(nn.Linear(output_dim, cfg.latent_dim))

    def forward(self, x):
        x = F.relu(self.conv(x / 255.0))
        return F.relu(self.fc1(x))


class WorldModel(nn.Module):
    def __init__(self, envs, cfg: AgentConfig):
        super().__init__()
        self.cfg = cfg
        obs_shape = envs.single_observation_space.shape
        act_dim = envs.single_action_space.n

        self._encoder = Encoder(cfg=cfg, envs=envs)
        self._trans = mlp(self.cfg.latent_dim + act_dim, cfg.mlp_dims, cfg.latent_dim)
        self._reward = mlp(self.cfg.latent_dim + act_dim, cfg.mlp_dims, 1)
        if cfg.use_projection:
            self._proj = mlp(self.cfg.latent_dim, cfg.mlp_dims, self.cfg.latent_dim)

        if cfg.compile:
            self._encoder = torch.compile(self._encoder, mode="default")
            self._trans = torch.compile(self._trans, mode="default")
            self._reward = torch.compile(self._reward, mode="default")

    def forward(self, obs, actions):
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        elif actions.ndim == 3:
            pass
        else:
            raise NotImplementedError
        horizon, batch_size, act_dim = actions.size()

        # Encoder observation
        zs = torch.empty(
            horizon + 1,
            batch_size,
            self.cfg.latent_dim,
            device=self.cfg.device,
        )
        zs[0] = self.encode(obs)

        # Rollout action sequence
        for t, action in enumerate(actions):
            zs[t + 1] = self.trans(zs[t], action)

        return zs

    def encode(self, obs):
        x = obs / 255.0
        if x.ndim == 5:  # [T, B, C, H, W]
            t, b = x.shape[:2]
            x = rearrange(x, "t b c h w -> (t b) c h w")
            z = enc_fn(x)
            z = rearrange(z, "(t b) d -> t b d", t=t, b=b)
        elif x.ndim == 4:
            z = enc_fn(x)
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")
        return z

    def trans(self, z, a):
        za = torch.concat([z, a], -1)
        delta_z = self._trans(za)
        next_z = z + delta_z if self.cfg.use_delta else delta_z
        return next_z

    def reward(self, z, a) -> torch.Tensor:
        za = torch.concat([z, a], -1)
        return self._reward(za)


# ALGO LOGIC: initialize agent here:
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder
class SoftQNetwork(nn.Module):
    def __init__(self, envs, cfg: AgentConfig):
        super().__init__()
        act_dim = envs.single_action_space.n
        self.mlp = mlp(cfg.latent_dim, cfg.mlp_dims, act_dim)

    def forward(self, z):
        # za = torch.concat([z, a], -1)
        q_vals = self.mlp(z)
        return q_vals


class Actor(nn.Module):
    def __init__(self, envs, cfg: AgentConfig):
        super().__init__()
        act_dim = envs.single_action_space.n
        self.mlp = mlp(cfg.latent_dim, cfg.mlp_dims, act_dim)

    def forward(self, z):
        return self.mlp(z)  # logits [B, A]

    # @torch.no_grad()
    def _infer_probs(self, logits):
        probs = F.softmax(logits, dim=-1).clamp_min(1e-8)
        log_pi = torch.log(probs)
        return probs, log_pi

    def get_action(self, z):
        logits = self.forward(z)
        probs, log_pi = self._infer_probs(logits)
        policy = Categorical(probs=probs)
        actions = policy.sample()  # [B]
        return {"actions": actions.long(), "probs": probs, "log_pi": log_pi}


class Agent:
    def __init__(self, cfg: AgentConfig, envs: gym.vector.SyncVectorEnv, device: torch.device):
        self.cfg = cfg
        self.act_dim = envs.single_action_space.n
        self.device = device

        self.model = WorldModel(envs, cfg=cfg).to(device)
        self.actor = Actor(envs, cfg).to(device)
        self.q1 = SoftQNetwork(envs, cfg).to(device)
        self.q2 = SoftQNetwork(envs, cfg).to(device)
        self.q1_target = SoftQNetwork(envs, cfg).to(device)
        self.q2_target = SoftQNetwork(envs, cfg).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
        self.model_optimizer = optim.AdamW(self.model.parameters(), lr=cfg.model_lr, eps=1e-4, weight_decay=0.0)
        self.q_optimizer = optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.q_lr, eps=1e-4, weight_decay=0.0
        )
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=cfg.policy_lr, eps=1e-4, weight_decay=0.0)

        # Automatic entropy tuning
        if cfg.autotune:
            self.target_entropy = -cfg.target_entropy_scale * torch.log(1 / torch.tensor(self.act_dim))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp().item()
            self.a_optimizer = optim.AdamW([self.log_alpha], lr=cfg.q_lr, eps=1e-4, weight_decay=0.0)
        else:
            self.alpha = cfg.alpha

    def _one_hot(self, a_idx: torch.Tensor) -> torch.Tensor:
        # a_idx: [T,B] or [B]
        shape = a_idx.shape
        oh = F.one_hot(a_idx.long(), num_classes=self.act_dim).float()
        return oh.view(*shape, self.act_dim)

    def update_step(self, batch: ReplayBufferTrajSamples, global_step: int) -> dict[str, Any]:
        info = {}

        # Update world model
        # info, zs = self.model_update_step(batch)
        m_info, zs = self.model_update_step(batch)
        info.update({f"model/{k}": v for k, v in m_info.items()})

        # Create a latent state-action transition for the actor/critic
        latent_batch = ReplayBufferLatentTrajSamples(
            zs=zs[0].detach(),
            actions=batch.actions[0],
            next_zs=zs[1].detach(),
            dones=batch.dones[0],
            rewards=batch.rewards[0],
            nstep_returns=batch.nstep_returns,
            discounts=batch.discounts[0],
        )

        # Update critic
        # info.update(self.critic_update_step(latent_batch, global_step=global_step))
        c_info = self.critic_update_step(latent_batch, global_step)
        info.update({f"critic/{k}": v for k, v in c_info.items()})

        # info.update(self.actor_update_step(zs[0].detach()))
        a_info = self.actor_update_step(zs[0].detach())
        info.update({f"actor/{k}": v for k, v in a_info.items()})

        return info
        # return {k: (v.item() if torch.is_tensor(v) and v.numel()==1 else v) for k,v in info.items()}

    def model_update_step(self, batch: ReplayBufferTrajSamples) -> Tuple[dict[str, Any], torch.Tensor]:
        loss, zs, info = self.model_loss(batch)
        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()
        return info, zs

    def critic_update_step(self, batch: ReplayBufferLatentTrajSamples, global_step: int) -> dict[str, Any]:
        loss, info = self.critic_loss(batch, global_step)
        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()
        return info

    def actor_update_step(self, zs: torch.Tensor) -> dict[str, Any]:
        loss, info = self.actor_loss(zs)
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return info

    def model_loss(self, batch: ReplayBufferTrajSamples):
        device = self.device
        T = self.cfg.horizon

        # encode targets
        with torch.no_grad():
            z_tar = self.model.encode(batch.next_observations.to(device))

        # rollout in latent space
        z0 = self.model.encode(batch.observations[0].to(device))  # [T,B,...]
        zs = [z0]
        a_idx = batch.actions[:T, :, 0].to(device)  # [T,B]
        a_oh = self._one_hot(a_idx)  # [T,B,A]
        for t in range(T):
            zs.append(self.model.trans(zs[t], a_oh[t]))
        zs = torch.stack(zs, dim=0)  # [T+1,B,Z]

        rho = torch.tensor([self.cfg.horizon and self.cfg.rho**t for t in range(T)], device=device, dtype=torch.float32)
        dones = batch.dones[:T].to(device).float()  # [T,B]
        rewards = batch.rewards[:T].to(device).float()

        # reward loss
        r_pred = self.model.reward(zs[:-1], a_oh)[..., 0]  # [T,B]
        r_mse = (r_pred - rewards) ** 2
        reward_loss = (rho[:, None] * ((1.0 - dones) * r_mse).mean(dim=1)).mean()

        if self.cfg.use_projection:
            zs_proj = self.model._proj(zs)
            z_tar_proj = self.model._proj(z_tar)
        else:
            zs_proj = zs
            z_tar_proj = z_tar
        if self.cfg.consistency_loss == "mse":
            # temporal consistency (MSE)
            tc = ((zs_proj[1:] - z_tar_proj) ** 2).mean(dim=-1)  # [T,B]
            tc_loss = (rho[:, None] * ((1.0 - dones) * tc).mean(dim=1)).mean()
        elif self.cfg.consistency_loss == "cosine":
            # temporal consistency (cosine similarity)
            tc = -nn.CosineSimilarity(dim=-1, eps=1e-6)(zs_proj[1:], z_tar_proj)
            tc_loss = (rho[:, None] * ((1.0 - dones) * tc).mean(dim=1)).mean()
        elif self.cfg.consistency_loss == "infonce":
            if self.cfg.use_horizon_as_negatives:
                zs_enc = rearrange(z_tar_proj, "h b d -> (h b) d")
                zs_dyn = rearrange(zs_proj[1:], "h b d -> (h b) d")
                tc = InfoNCE()(zs_dyn, zs_enc)
            else:
                tc = torch.vmap(InfoNCE())(zs_proj[1:], z_tar_proj)
                # TODO does this need to consider dones?
            tc_loss = (tc * rho).mean()
        else:
            raise NotImplementedError

        loss = self.cfg.consistency_coef * tc_loss + self.cfg.reward_coef * reward_loss

        return (
            loss,
            zs,
            {
                "model_loss": loss.detach(),
                "tc_loss": tc_loss.detach(),
                "reward_loss": reward_loss.detach(),
                "z_mean": zs.float().mean().detach(),
                "z_min": torch.min(zs).detach(),
                "z_max": torch.max(zs).detach(),
                "r_mean": r_pred.mean().detach(),
                "r_min": r_pred.min().detach(),
                "r_max": r_pred.max().detach(),
            },
        )

    def critic_loss(self, batch: ReplayBufferLatentTrajSamples, global_step: int) -> Tuple[torch.Tensor, dict[str, Any]]:
        # CRITIC training
        with torch.no_grad():
            next_actions = self.actor.get_action(batch.next_zs)
            q1_next_target = self.q1_target(batch.next_zs)
            q2_next_target = self.q2_target(batch.next_zs)
            # we can use the action probabilities instead of MC sampling to estimate the expectation
            min_q_next_target = next_actions["probs"] * (
                torch.min(q1_next_target, q2_next_target) - self.alpha * next_actions["log_pi"]
            )
            # adapt Q-target for discrete Q-function
            min_q_next_target = min_q_next_target.sum(dim=1)
            next_q_value = batch.rewards.flatten() + (1 - batch.dones.flatten()) * self.cfg.gamma * (min_q_next_target)

        # use Q-values only for the taken actions
        q1_values = self.q1(batch.zs)
        q2_values = self.q2(batch.zs)
        q1_a_values = q1_values.gather(1, batch.actions.long()).view(-1)
        q2_a_values = q2_values.gather(1, batch.actions.long()).view(-1)
        q1_loss = F.mse_loss(q1_a_values, next_q_value)
        q2_loss = F.mse_loss(q2_a_values, next_q_value)
        q_loss = q1_loss + q2_loss
        info = {
            "q1_values": q1_a_values.mean().detach(),
            "q2_values": q2_a_values.mean().detach(),
            "q1_loss": q1_loss.detach(),
            "q2_loss": q2_loss.detach(),
            "q_loss": q_loss.detach(),
        }

        # update the target networks
        if global_step % self.cfg.target_network_frequency == 0:
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.data.copy_(self.cfg.tau * param.data + (1 - self.cfg.tau) * target_param.data)

        return q_loss, info

    def actor_loss(self, zs: torch.Tensor) -> Tuple[torch.Tensor, dict[str, Any]]:
        # ACTOR training
        actions = self.actor.get_action(zs)
        with torch.no_grad():
            q1_values = self.q1(zs)
            q2_values = self.q2(zs)
            min_q_values = torch.min(q1_values, q2_values)
        # no need for reparameterization, the expectation can be calculated for discrete actions
        actor_loss = (actions["probs"] * ((self.alpha * actions["log_pi"]) - min_q_values)).mean()

        if self.cfg.autotune:
            # reuse action probabilities for temperature loss
            alpha_loss = (
                actions["probs"].detach() * (-self.log_alpha.exp() * (actions["log_pi"] + self.target_entropy).detach())
            ).mean()

            self.a_optimizer.zero_grad()
            alpha_loss.backward()
            self.a_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        info = {
            "actor_loss": actor_loss.detach(),
            "alpha": self.alpha,
        }
        if self.cfg.autotune:
            info.update({"alpha_loss": alpha_loss.detach()})
        return actor_loss, info

    def get_action(self, x):
        z = self.model.encode(x)
        return self.actor.get_action(z)


# if __name__ == "__main__":
def main(cfg):
    """Main training function."""
    # print_header("🚀 JAX-RL Training")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.cuda else "cpu")

    # Print configuration
    # print_section("Configuration")
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
        # print_section("Initializing Wandb")

        wandb_run = wandb.init(
            project=cfg.wandb_project_name,
            entity=cfg.wandb_entity,
            sync_tensorboard=True,
            config=asdict(cfg),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        # print_success(f"Wandb run: {wandb_run.name}")
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
    # print_section("Creating Environment")
    envs = gym.vector.SyncVectorEnv([make_env(cfg.env_id, cfg.seed, 0, cfg.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # Create agent
    # print_section("Creating Agent")
    agent = Agent(cfg.agent, envs=envs, device=device)

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
            actions = agent.get_action(torch.Tensor(obs).to(device))
            actions = actions["actions"].detach().cpu().numpy()

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
                batch = rb.sample(cfg.agent.batch_size, cfg.agent.horizon)
                info = agent.update_step(batch, global_step=global_step)

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

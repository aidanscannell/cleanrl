#!/usr/bin/env python3
import numpy as np
import torch


def evaluate(cfg, agent, eval_envs, global_step: int, writer):
    obs, info = eval_envs.reset()

    # num_envs = getattr(eval_envs, "num_envs", len(getattr(eval_envs, "env_fns", [])) or 1)
    episode_returns = np.zeros(cfg.num_eval_episodes, dtype=np.float64)
    episode_lengths = np.zeros(cfg.num_eval_episodes, dtype=np.int32)
    done_flag = np.zeros(cfg.num_eval_episodes, dtype=bool)

    finished_returns = []
    finished_lengths = []

    # ---- rollout: one episode per env --------------------------------------
    while not np.all(done_flag):
        # Get greedy/deterministic actions for all active envs
        actions = agent.get_action(torch.Tensor(obs).to(agent.device), eval_mode=True)

        # Step the vector env
        obs, reward, terminated, truncated, info = eval_envs.step(actions["actions"].cpu().numpy())
        done = np.logical_or(terminated, truncated)

        # Accumulate rewards/lengths only for not-yet-finished envs
        episode_returns += reward * (~done_flag)
        episode_lengths += (~done_flag).astype(np.int32)

        # For envs that just finished now, store and mark done
        just_finished = (~done_flag) & done
        if np.any(just_finished):
            finished_returns.extend(episode_returns[just_finished].tolist())
            finished_lengths.extend(episode_lengths[just_finished].tolist())
            done_flag[just_finished] = True

    # ---- summarize ----------------------------------------------------------
    finished_returns = np.asarray(finished_returns, dtype=np.float64)
    finished_lengths = np.asarray(finished_lengths, dtype=np.int32)

    results = {
        "episodic_return": float(finished_returns.mean()) if len(finished_returns) else 0.0,
        "episodic_return_std": float(finished_returns.std(ddof=0)) if len(finished_returns) else 0.0,
        "episodic_length": float(finished_lengths.mean()) if len(finished_lengths) else 0.0,
        "episodic_length_std": float(finished_lengths.std(ddof=0)) if len(finished_lengths) else 0.0,
        "global_step": global_step,
    }
    for key, value in results.items():
        writer.add_scalar(f"eval/{key}", value, global_step)

    print(
        f"[EVAL] Step: {global_step} | Return: {results['episodic_return']:.2f} ± {results['episodic_return_std']:.2f} | "
        f"Length: {int(results['episodic_length'])} ± {int(results['episodic_length_std'])}"
    )
    return results

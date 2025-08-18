"""Logging utilities for JAX-RL."""

import os
import time
from typing import Any, Dict, Optional, Union

import numpy as np
import wandb
from termcolor import colored


# Color codes for beautiful terminal output
class Colors:
    """ANSI color codes for terminal output."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Bright colors
    BRIGHT_RED = "\033[91;1m"
    BRIGHT_GREEN = "\033[92;1m"
    BRIGHT_YELLOW = "\033[93;1m"
    BRIGHT_BLUE = "\033[94;1m"
    BRIGHT_MAGENTA = "\033[95;1m"
    BRIGHT_CYAN = "\033[96;1m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_RED = "\033[41m"


def colored_print(text: str, color: str, bold: bool = False, end: str = "\n") -> None:
    """Print colored text to console."""
    bold_code = Colors.BOLD if bold else ""
    print(f"{color}{bold_code}{text}{Colors.ENDC}", end=end)


def print_header(text: str) -> None:
    """Print a beautiful header."""
    print()
    colored_print("=" * 60, Colors.BRIGHT_BLUE, bold=True)
    colored_print(f" {text} ", Colors.BRIGHT_BLUE, bold=True)
    colored_print("=" * 60, Colors.BRIGHT_BLUE, bold=True)
    print()


def print_section(text: str) -> None:
    """Print a section header."""
    colored_print(f"▶ {text}", Colors.BRIGHT_CYAN, bold=True)


def print_success(text: str) -> None:
    """Print success message."""
    colored_print(f"✅ {text}", Colors.BRIGHT_GREEN)


def print_info(text: str) -> None:
    """Print info message."""
    colored_print(f"ℹ️  {text}", Colors.BRIGHT_BLUE)


def print_warning(text: str) -> None:
    """Print warning message."""
    colored_print(f"⚠️  {text}", Colors.BRIGHT_YELLOW)


def print_error(text: str) -> None:
    """Print error message."""
    colored_print(f"❌ {text}", Colors.BRIGHT_RED)


def print_progress(current: int, total: int, prefix: str = "Progress", width: int = 30) -> None:
    """Print a beautiful progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    colored_print(f"{prefix}: |{bar}| {current}/{total} ({percent:.1%})", Colors.BRIGHT_GREEN)


def print_metrics(metrics: Dict[str, Any], prefix: str = "") -> None:
    """Print metrics in a beautiful format."""
    if prefix:
        colored_print(f"{prefix}:", Colors.BRIGHT_CYAN, bold=True)

    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        colored_print(f"  {key}: {formatted_value}", Colors.OKCYAN)


def print_eval_summary(step: int, length: int, reward: float, success: bool = False) -> None:
    """Print evaluation episode summary."""
    status = "🎯 SUCCESS" if success else "📊 COMPLETED"
    colored_print(
        f"Step {step} {status}: {length} steps, reward: {reward:.3f}", Colors.BRIGHT_GREEN if success else Colors.OKGREEN
    )


def print_video_info(frames: int, resolution: tuple, fps: int) -> None:
    """Print video recording information."""
    colored_print(f"🎬 Video recorded: {frames} frames, {resolution[0]}x{resolution[1]}, {fps} FPS", Colors.BRIGHT_MAGENTA)


def print_run(cfg, env):
    """Print information about run"""
    # Create a border
    border = "=" * 50

    def print_aligned(key, value, color="green"):
        key = key + ":"
        print(colored(f"  {key:<20}", color, attrs=["bold"]), f"{value:<30}")

    task = cfg.env_name if cfg.task_name == "" else cfg.env_name + "-" + cfg.task_name
    obs_spec = env.observation_spec["observation"][0]
    act_spec = env.action_spec[0]

    data = [
        ("Task", task),
        ("Steps", f"{(cfg.num_episodes * cfg.max_episode_steps) / 1e6}M"),
        ("Episodes", cfg.num_episodes),
        ("Observations", np.array(obs_spec["state"].shape).prod().item()),
        ("Actions", np.array(act_spec.shape).prod().item()),
        ("Action repeat", cfg.action_repeat),
        ("Device", cfg.device),
    ]

    print(f"\n{border}")

    for row in data:
        print_aligned(row[0], row[1], color="green")

    print(f"{border}")


def print_metrics(step: int, metrics: dict, eval_mode: bool = False):
    prefix = "Eval" if eval_mode else "Train"
    color = "green" if eval_mode else "blue"
    episode_return = f"{metrics['episode']['r']:.2f}"
    if "success" in metrics.keys():
        # TODO read success correctly
        success = metrics["success"]
        m = f"{prefix:<7} E: {step:<10}  R: {episode_return:<14} S: {success:<16}"
    else:
        m = f"{prefix:<7} E: {step:<10}  R: {episode_return:<14}"
    print(colored(m, color))

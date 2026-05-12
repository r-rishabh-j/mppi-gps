"""Plot GPS training curves, ablations, and SB3 baseline comparisons."""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(curves_path: Path, save_dir: Path):
    """Plot GPS training curves: episode cost over iterations."""
    data = json.loads(curves_path.read_text())
    env_name = curves_path.stem.replace("gps_", "").replace("_curves", "")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # Accept legacy "costs" key.
    cost_key = "mppi_costs" if "mppi_costs" in data else "costs"
    ax.plot(data[cost_key])
    ax.set_xlabel("GPS iteration")
    ax.set_ylabel("mean episode cost")
    ax.set_title(f"GPS Training — {env_name}")

    plt.tight_layout()
    out = save_dir / f"gps_{env_name}_training.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close()


def plot_ablations(ablation_path: Path, save_dir: Path):
    """Plot ablation results: one figure per ablation axis."""
    data = json.loads(ablation_path.read_text())
    env_name = ablation_path.stem.replace("_ablations", "")

    # ---- Alpha ablation: bar chart of final GPS cost for each alpha value ----
    if "alpha" in data:
        fig, ax = plt.subplots(figsize=(6, 4))
        alphas = sorted(data["alpha"].keys(), key=float)
        costs = [data["alpha"][a]["gps_mean_cost"] for a in alphas]
        # Show MPPI baseline as a horizontal reference line
        mppi_cost = data["alpha"][alphas[0]]["mppi_mean_cost"]
        ax.bar(range(len(alphas)), costs, tick_label=[f"a={a}" for a in alphas])
        ax.axhline(mppi_cost, color="r", linestyle="--", label="MPPI")
        ax.set_ylabel("mean cost")
        ax.set_title(f"Alpha ablation — {env_name}")
        ax.legend()
        plt.tight_layout()
        out = save_dir / f"{env_name}_ablation_alpha.png"
        plt.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close()

    # ---- K ablation: overlaid cost curves for different sample counts ----
    if "K" in data:
        fig, ax = plt.subplots(figsize=(6, 4))
        for k_val, result in data["K"].items():
            ax.plot(result["costs_curve"], label=f"K={k_val}")
        ax.set_xlabel("GPS iteration")
        ax.set_ylabel("mean cost")
        ax.set_title(f"Sample count ablation — {env_name}")
        ax.legend()
        plt.tight_layout()
        out = save_dir / f"{env_name}_ablation_K.png"
        plt.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close()

    # ---- Num conditions ablation: bar chart ----
    if "num_conditions" in data:
        fig, ax = plt.subplots(figsize=(6, 4))
        ncs = sorted(data["num_conditions"].keys(), key=int)
        costs = [data["num_conditions"][nc]["gps_mean_cost"] for nc in ncs]
        ax.bar(range(len(ncs)), costs, tick_label=[f"N={nc}" for nc in ncs])
        ax.set_ylabel("mean cost")
        ax.set_title(f"Num conditions ablation — {env_name}")
        plt.tight_layout()
        out = save_dir / f"{env_name}_ablation_conditions.png"
        plt.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close()

    # ---- Wall-clock: two bars (policy vs MPPI) ----
    if "wallclock" in data:
        wc = data["wallclock"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Policy", "MPPI"], [wc["policy_ms"], wc["mppi_ms"]])
        ax.set_ylabel("time per step (ms)")
        ax.set_title(f"Wall-clock comparison — {env_name}\n({wc['speedup']:.0f}x speedup)")
        plt.tight_layout()
        out = save_dir / f"{env_name}_wallclock.png"
        plt.savefig(out, dpi=150)
        print(f"saved {out}")
        plt.close()


def plot_comparison_bar(results_dir: Path, save_dir: Path, env_name: str):
    """Bar chart comparing final cost across all available methods.

    Looks for GPS, MPPI, SAC, and PPO results and plots whatever is available.
    """
    methods = {}

    # GPS — last iteration cost from the training curves
    gps_curves = results_dir / f"gps_{env_name}_curves.json"
    if gps_curves.exists():
        d = json.loads(gps_curves.read_text())
        cost_key = "mppi_costs" if "mppi_costs" in d else "costs"
        methods["GPS"] = d[cost_key][-1] if d.get(cost_key) else None

    # MPPI — from the ablation file (uses the first alpha's MPPI eval)
    abl_path = results_dir / f"{env_name}_ablations.json"
    if abl_path.exists():
        d = json.loads(abl_path.read_text())
        if "alpha" in d:
            first_key = list(d["alpha"].keys())[0]
            methods["MPPI"] = d["alpha"][first_key]["mppi_mean_cost"]

    # SB3 baselines — look for SAC and PPO result files
    for algo in ["sac", "ppo"]:
        sb3_path = results_dir / "sb3" / f"{algo}_{env_name}_results.json"
        if sb3_path.exists():
            d = json.loads(sb3_path.read_text())
            methods[algo.upper()] = d.get("mean_cost")

    if len(methods) < 2:
        print(f"not enough methods found for comparison bar chart ({env_name})")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(methods.keys())
    values = [methods[n] for n in names]
    ax.bar(range(len(names)), values, tick_label=names)
    ax.set_ylabel("mean cost")
    ax.set_title(f"Method comparison — {env_name}")
    plt.tight_layout()
    out = save_dir / f"{env_name}_comparison.png"
    plt.savefig(out, dpi=150)
    print(f"saved {out}")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="acrobot")
    p.add_argument("--results-dir", type=str, default="results/ablations",
                   help="Directory containing ablation JSON files")
    p.add_argument("--curves-dir", type=str, default="checkpoints",
                   help="Directory containing GPS curves JSON files")
    p.add_argument("--save-dir", type=str, default="results/plots")
    return p.parse_args()


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot GPS training curves (if available)
    curves_path = Path(args.curves_dir) / f"gps_{args.env}_curves.json"
    if curves_path.exists():
        plot_training_curves(curves_path, save_dir)

    # Plot ablation results (if available)
    abl_path = Path(args.results_dir) / f"{args.env}_ablations.json"
    if abl_path.exists():
        plot_ablations(abl_path, save_dir)

    # Plot cross-method comparison bar chart
    plot_comparison_bar(Path(args.curves_dir), save_dir, args.env)


if __name__ == "__main__":
    main()

"""Plot the prespecified development points; never interpolate missing endpoints."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1])
OUT = parser.parse_args().output_root.resolve()
data = json.loads((OUT / "results_summary.json").read_text())
evaluation = data["evaluation"]
styles = {
    "uniform": ("Uniform-R2", "#737373", "o"),
    "sr": ("SR-gap sync-R2", "#0072B2", "s"),
    "residual": ("MSE-R2 (Ada/micro8)", "#D55E00", "^"),
    "residual_matched": ("MSE-R2 (matched Blackwell)", "#009E73", "D"),
}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
for method, (label, color, marker) in styles.items():
    points = [(0, evaluation["initial400"])] + [
        (r, evaluation[f"{method}_round{r}"])
        for r in (10, 30) if f"{method}_round{r}" in evaluation
    ]
    pending = not data["methods"][method]["complete_curve"]
    for ax, metric in zip(axes, ("macro_sr", "max_teacher_gap")):
        ax.plot(
            [r for r, _ in points], [100 * point[metric] for _, point in points],
            label=label + (" (pending)" if pending else ""), color=color,
            marker=marker, linewidth=1.8, linestyle="--" if pending else "-",
        )
for ax, title in zip(axes, ("Macro success (higher is better)", "Max teacher gap (lower is better)")):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Training rollout rounds (32 trajectories / round)")
    ax.set_ylabel("Percentage points" if ax is axes[1] else "Success rate (%)")
    ax.set_xlim(-1, 31)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_ylim(0, 100)
    ax.grid(alpha=.2)
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Development probe: seed0, 20 episodes / task; Ada/micro8 run is supplementary", fontsize=10)
fig.savefig(OUT / "development_curves.png", dpi=180)
fig.savefig(OUT / "development_curves.pdf")
plt.close(fig)

tasks = data["tasks"]
fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
series = [("initial400", "SFT400", "#BBBBBB")] + [
    (f"{method}_round30", label, color)
    for method, (label, color, _) in styles.items() if f"{method}_round30" in evaluation
]
width = .72 / len(series)
for j, (key, label, color) in enumerate(series):
    ax.bar([i - .36 + width * (j + .5) for i in range(4)],
           [100 * evaluation[key]["task_sr"][t] for t in tasks],
           width, label=label, color=color)
ax.scatter(range(4), [100 * evaluation["teachers"]["task_sr"][t] for t in tasks],
           marker="_", s=500, linewidths=2, color="black", label="Teacher")
ax.set_xticks(range(4), ["Open laptop", "Handover mic", "Dump bin", "Stack bowls"])
ax.set_ylabel("Success rate (%)")
ax.set_ylim(0, 108)
ax.set_title("Round30 development endpoint (seed0, 20 episodes / task)")
ax.grid(axis="y", alpha=.2)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(.5, -.12))
fig.savefig(OUT / "development_endpoint.png", dpi=180)
fig.savefig(OUT / "development_endpoint.pdf")
plt.close(fig)
print("Saved development curves and endpoint PNG/PDF artifacts.")

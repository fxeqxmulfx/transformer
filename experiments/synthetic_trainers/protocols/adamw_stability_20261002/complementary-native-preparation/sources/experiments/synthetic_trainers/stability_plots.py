"""Standalone calibration and collapse-neighbor figures; no PyTorch required."""

from .stability_analysis import diagnostic_metrics


def export(figure, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")


def render_run(report, probes, diagnostics, summary, directory, *, gradients=()):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history, config = report["history"], report["plan"]["config"]
    xs = [p["step"] for p in history]
    figure, axes = plt.subplots(3, 2, figsize=(12, 10.4), constrained_layout=True)
    for split, color in (("train", "#0072b2"), ("heldout", "#d55e00")):
        axes[0, 0].plot(xs, [p[split]["accuracy"] for p in history], label=split, color=color)
    axes[0, 0].scatter([p["step"] for p in probes], [p["heldout"]["accuracy"] for p in probes],
                       label="held-out neighbor probes", color="grey", alpha=.25, s=4)
    axes[0, 0].axhline(config["target"], color="black", linestyle=":", linewidth=.7)
    axes[0, 0].set(ylim=(-.02, 1.025), ylabel="Complete RHS accuracy")
    for metric, color in (("answer_loss", "#d55e00"), ("EOS_loss", "#009e73")):
        axes[0, 1].plot(xs, [max(1e-12, p["heldout"][metric]) for p in history], label=metric, color=color)
    axes[0, 1].set(yscale="log", ylabel="Held-out CE (display floor 1e-12 nats)")
    metrics = [diagnostic_metrics(row) for row in diagnostics]
    gradient_points = gradients or metrics
    for tail, color, label in ((False, "#0072b2", "ordinary/full batches"), (True, "#d55e00", "epoch tails")):
        selected = [row for row in gradient_points if row["epoch_tail"] == tail]
        axes[1, 0].scatter([p["step"] for p in selected], [p["gradient_l2"] for p in selected],
                           s=5, alpha=.5, label=label, color=color)
    axes[1, 0].set_yscale("symlog", linthresh=1e-8)
    axes[1, 0].set(ylabel="Pre-update joint gradient L2" + (" (every update)" if gradients else ""))
    moment_key = "maximum_l2" if config["optimizer"] == "amsgradw" else "exp_avg_sq_l2"
    for key, color in (("parameter_l2", "#0072b2"), ("update_l2", "#d55e00"), (moment_key, "#009e73")):
        selected = [p for p in metrics if p[key] > 0]
        axes[1, 1].plot([p["step"] for p in selected], [p[key] for p in selected], label=key, color=color, linewidth=1)
    axes[1, 1].set(yscale="log", ylabel="Post-update joint L2 norms")
    if diagnostics:
        for name in diagnostics[0]["temperatures"]:
            for head in range(len(diagnostics[0]["temperatures"][name]["inverse_temperature"])):
                axes[2, 0].plot([p["step"] for p in diagnostics],
                    [p["temperatures"][name]["inverse_temperature"][head] for p in diagnostics],
                    label=f"{name.split('.')[1]}:{head}", linewidth=.8)
    axes[2, 0].set(ylabel="Learned inverse temperature", xlabel="Updates")
    axes[2, 0].legend(title="Layer:head", fontsize=7, ncol=4)
    axes[2, 1].plot(xs, [p["heldout"]["answer_accuracy"] for p in history], label="numeric answer", color="#d55e00")
    axes[2, 1].plot(xs, [p["heldout"]["EOS_accuracy"] for p in history], label="EOS", color="#009e73")
    axes[2, 1].set(ylim=(-.02, 1.025), ylabel="Held-out component accuracy", xlabel="Updates")
    for index, axis in enumerate(axes.flat):
        axis.grid(alpha=.2)
        axis.axvspan(max(0, config["steps"] - summary["assessment"]["criterion"]["tail_steps"]),
                    config["steps"], color="grey", alpha=.055)
        if index != 4:
            axis.legend(fontsize=8)
    kind = "confirmation" if summary["scope"].startswith("one_completed_independent_confirmation_run") else "calibration"
    optimizer = {"amsgradw": "raw AMSGradW", "adamw": "AdamW"}.get(config["optimizer"], config["optimizer"])
    figure.suptitle(f"{summary['name']}: {optimizer} / GPTMini\n"
                   f"lr={config['learning_rate']:g}, {config['batch_policy']}; complete {config['steps']:,}-update {kind}\n"
                   "Gray band: prospectively scored final tail", fontsize=12)
    export(figure, directory, "stability-overview")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.3), constrained_layout=True)
    neighborhoods = summary["collapse_diagnostics"]["neighborhoods"]
    for name, color, marker in (("before", "#0072b2", "<"), ("canonical", "#d55e00", "o"), ("after", "#009e73", ">")):
        axes[0].scatter([p["step"] for p in neighborhoods], [p["heldout"][name]["accuracy"] for p in neighborhoods],
                        s=10, alpha=.6, label=name, color=color, marker=marker)
    axes[0].axhline(config["target"], color="black", linestyle=":", linewidth=.7)
    axes[0].set(ylim=(-.02, 1.025), xlabel="Canonical failed observation", ylabel="Held-out complete RHS accuracy")
    supported = [p for p in neighborhoods if all(p["diagnostics"].values())]
    if supported:
        axes[1].scatter([p["diagnostics"]["canonical"]["gradient_l2"] for p in supported],
                        [p["heldout"]["canonical"]["accuracy"] for p in supported], s=12, color="#d55e00", alpha=.6)
        axes[1].set_xscale("log")
    axes[1].set(ylim=(-.02, 1.025), xlabel="Pre-update joint gradient L2 at canonical step", ylabel="Held-out complete RHS accuracy")
    for axis in axes:
        axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Immediate neighbors of later canonical failures: {len(neighborhoods)} supported triplets\n"
                   "Descriptions of observed trajectories; association does not establish a cause", fontsize=12)
    export(figure, directory, "collapse-neighbors")
    plt.close(figure)

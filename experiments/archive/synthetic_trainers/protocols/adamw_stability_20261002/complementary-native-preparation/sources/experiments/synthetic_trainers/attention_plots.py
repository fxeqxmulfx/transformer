"""Complete paired curves and all nominal post-onset recovery windows."""

import json

from .stability_plots import export


def render_pair(directory, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    colors={"adamw-softmax":"#0072b2","adamw-sparsemax":"#d55e00"}
    window_count,episode_count=0,0
    for name,color in colors.items():
        report=json.loads((directory/"runs"/name/"measurements.json").read_text())
        points=report["history"]
        metrics=result["recovery"][name]["tail_and_episodes"]
        series=result["recovery"][name]["whole_post_onset_series"]
        x=[p["step"]/1000 for p in points]
        axes[0,0].plot(x,[p["heldout"]["accuracy"] for p in points],label=name,color=color)
        axes[0,0].plot(x,[p["train"]["accuracy"] for p in points],ls=":",color=color,alpha=.65,label=name+" train")
        axes[1,1].plot(x,[max(p["heldout"]["answer_loss"],1e-12) for p in points],color=color,label=name)
        axes[1,2].plot([p["training_seconds"] for p in points],[p["heldout"]["accuracy"] for p in points],color=color,label=name)
        if series["metrics"] is not None:
            windows=[w for w in series["metrics"]["joint"]["fixed_grid_windows"] if w["complete_window"] and w["nominal_width_window"]]
            window_count+=len(windows)
            centers=[(w["start"]+w["stop"])/2000 for w in windows]
            label=name if windows else name+": no full post-onset window"
            axes[0,1].plot(centers,[100*w["failure_fraction"] for w in windows],"o-",ms=3,color=color,label=label)
            axes[0,2].plot(centers,[w["episode_onsets_per_10000_updates"] for w in windows],"o-",ms=3,color=color,label=label)
        else:
            for axis in (axes[0,1],axes[0,2]):
                axis.plot([],[],color=color,label=name+": no long onset")
        joint=metrics["metrics"]["joint"]
        episodes=joint["episodes_after_long_onset"]
        episode_count+=len(episodes)
        if not episodes:
            label=name+(": no long onset" if joint["episode_count"] is None else ": zero post-onset episodes")
            axes[1,0].plot([],[],color=color,label=label)
        for episode in episodes:
            duration=episode["sampled_steps_until_first_recovery"]
            if duration is None:
                axes[1,0].scatter(episode["first_failure"]/1000,metrics["through_update"]-episode["first_failure"],
                                  marker=">",color=color,label=name+" censored at budget")
            else:
                axes[1,0].scatter(episode["first_failure"]/1000,duration,color=color,label=name)
    if not window_count:
        for axis in (axes[0,1],axes[0,2]):
            axis.text(.5,.5,"Post-onset window rates unavailable",ha="center",transform=axis.transAxes)
            axis.set_yticks([])
    if not episode_count:
        axes[1,0].text(.5,.5,"No sampled recovery episodes",ha="center",transform=axes[1,0].transAxes)
        axes[1,0].set_yticks([])
    axes[0,0].axhline(metrics["criterion"]["target"],ls="--",lw=.7,color="grey")
    axes[0,0].set(xlabel="Updates (thousands)",ylabel="Complete RHS accuracy",ylim=(-.02,1.025),title="Every canonical observation")
    axes[0,1].set(xlabel="Window midpoint (thousands of updates)",ylabel="Failed joint observations (%)",title="Fully observed 10,000-update windows")
    axes[0,2].set(xlabel="Window midpoint (thousands of updates)",ylabel="New episodes / 10,000 updates",title="Whole post-long-onset grid")
    axes[1,0].set(xlabel="First failed update (thousands)",ylabel="Sampled updates to first recovery",title="Recovery durations; > marks censoring")
    axes[1,1].set(xlabel="Updates (thousands)",ylabel="Held-out answer CE (display floor 1e-12)",yscale="log",title="Numeric answers scored separately")
    axes[1,2].set(xlabel="Measured training seconds",ylabel="Held-out complete RHS accuracy",ylim=(-.02,1.025),title="Quality versus measured update time")
    for axis in axes.flat:
        axis.grid(alpha=.2)
        handles,labels=axis.get_legend_handles_labels()
        if handles:
            unique=dict(zip(labels,handles))
            axis.legend(unique.values(),unique.keys(),fontsize=8)
    figure.suptitle("Native AdamW · causal softmax/sparsemax · one frozen initialization and split\n"
                   "Complete histories and descriptive sampled recovery; independent confirmations outstanding",fontsize=13)
    export(figure,directory/"plots","attention-recovery")
    plt.close(figure)

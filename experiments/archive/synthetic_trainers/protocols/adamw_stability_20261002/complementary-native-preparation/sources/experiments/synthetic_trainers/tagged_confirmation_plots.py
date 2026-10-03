"""All six tagged trajectories and sampled recovery, adapted from scheduled_plots.py."""

import json

from .stability_plots import export


def render_confirmation(directory,metrics,plan):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure,axes=plt.subplots(2,3,figsize=(17,8.5),constrained_layout=True)
    colors=("#0072b2","#d55e00","#009e73","#cc79a7","#e69f00","#56b4e9")
    windows_seen,episodes_seen=0,0
    for recipe,color in zip(plan["recipes"],colors):
        name=recipe["name"];path=directory/"runs"/name
        report=json.loads((path/"measurements.json").read_text());points=report["history"]
        x=[p["step"]/1000 for p in points]
        axes[0,0].plot(x,[p["heldout"]["accuracy"] for p in points],label=name,color=color)
        axes[0,0].plot(x,[p["train"]["accuracy"] for p in points],ls=":",alpha=.5,color=color)
        gradients=[json.loads(line) for line in (path/"gradients.jsonl").read_text().splitlines()]
        axes[0,1].plot([g["step"]/1000 for g in gradients],[g["learning_rate"] for g in gradients],label=name,color=color)
        axes[1,2].plot(x,[max(p["heldout"]["answer_loss"],1e-12) for p in points],label=name,color=color)
        data=metrics["recovery"][name];series=data["whole_post_onset_series"]
        if series["metrics"] is not None:
            windows=[w for w in series["metrics"]["joint"]["fixed_grid_windows"] if w["complete_window"] and w["nominal_width_window"]]
            windows_seen+=len(windows);centers=[(w["start"]+w["stop"])/2000 for w in windows]
            label=name if windows else name+": no full window"
            axes[0,2].plot(centers,[100*w["failure_fraction"] for w in windows],"o-",ms=3,label=label,color=color)
            axes[1,0].plot(centers,[w["episode_onsets_per_10000_updates"] for w in windows],"o-",ms=3,label=label,color=color)
        else:
            for axis in (axes[0,2],axes[1,0]):
                axis.plot([],[],label=name+": no long onset",color=color)
        tail=data["tail_and_episodes"];joint=tail["metrics"]["joint"]
        episodes=joint["episodes_after_long_onset"];episodes_seen+=len(episodes)
        if not episodes:
            axes[1,1].plot([],[],label=name+(": no long onset" if joint["episode_count"] is None else ": zero episodes"),color=color)
        for episode in episodes:
            duration=episode["sampled_steps_until_first_recovery"];censored=duration is None
            axes[1,1].scatter(episode["first_failure"]/1000,tail["through_update"]-episode["first_failure"] if censored else duration,
                marker=">" if censored else "o",label=name+(" censored" if censored else ""),color=color)
    if not windows_seen:
        for axis in (axes[0,2],axes[1,0]):
            axis.text(.5,.5,"Post-onset window rates unavailable",ha="center",transform=axis.transAxes);axis.set_yticks([])
    if not episodes_seen:
        axes[1,1].text(.5,.5,"No sampled recovery episodes",ha="center",transform=axes[1,1].transAxes);axes[1,1].set_yticks([])
    axes[0,0].axhline(plan["criterion"]["target"],ls="--",color="grey",lw=.7)
    axes[0,0].set(xlabel="Updates (thousands)",ylabel="Complete RHS accuracy (dotted: train)",ylim=(-.02,1.025),title="Every canonical observation; all six cases")
    axes[0,1].set(xlabel="Updates (thousands)",ylabel="Actual learning rate",title="Every native optimizer update")
    axes[0,2].set(xlabel="Window midpoint (thousands of updates)",ylabel="Failed joint observations (%)",title="Fully observed 10,000-update windows")
    axes[1,0].set(xlabel="Window midpoint (thousands of updates)",ylabel="New episodes / 10,000 updates",title="Whole post-long-onset grid")
    axes[1,1].set(xlabel="First failed update (thousands)",ylabel="Sampled updates to first recovery",title="Recovery durations; > marks censoring")
    axes[1,2].set(xlabel="Updates (thousands)",ylabel="Held-out answer CE (display floor 1e-12)",yscale="log",title="Numeric answers scored separately")
    for axis in axes.flat:
        axis.grid(alpha=.2);handles,labels=axis.get_legend_handles_labels()
        if handles:
            unique=dict(zip(labels,handles));axis.legend(unique.values(),unique.keys(),fontsize=6.5)
    scope="Six scientific crossed confirmations" if plan["scientific_run"] else "Explicit CPU pipeline fixture; no scientific learning claim"
    figure.suptitle("Native AdamW · unchanged softmax GPTMini · every frozen case retained\n"+scope,fontsize=13)
    export(figure,directory/"plots","confirmation-recovery");plt.close(figure)

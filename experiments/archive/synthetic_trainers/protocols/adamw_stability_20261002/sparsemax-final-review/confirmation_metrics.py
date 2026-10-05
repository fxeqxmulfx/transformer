"""Step-5 metrics from canonical mod-193 observations, independent of the lab's two-point transition.

Policy: EXPERIMENT_PLAN.md, step 5, and the prospective criteria in
experiments/archive/synthetic_trainers/STABILITY.md: a pre-target plateau
of five observations spanning 1,000 updates, 20 joint target observations,
and the complete final 50,000-update window. A confirmed repair ends at
100% on both splits; persistence and a memorization plateau are reported
separately, as the softmax reference itself can fail persistence.
"""

import math


def measured_history(history, budget=300_000, cadence=250):
    """Require the complete regular history, excluding separately stored neighbor probes."""
    if [point['step'] for point in history] != list(range(0, budget + 1, cadence)):
        raise ValueError('The canonical history does not cover the complete budget at its cadence')
    for point in history:
        if point.get('diagnostic_probe', False):
            raise ValueError('A neighbor probe entered the canonical history')
        for split in ('train', 'heldout'):
            value = point[split]['accuracy']
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid {split} accuracy at {point["step"]}')
    return history


def blocks(points, predicate):
    found, current = [], []
    for point in points:
        if predicate(point):
            current.append(point)
        else:
            if current:
                found.append(current)
            current = []
    if current:
        found.append(current)
    return found


def summarize(history, budget=300_000, cadence=250):
    history = measured_history(history, budget, cadence)
    joint = lambda point: min(point['train']['accuracy'], point['heldout']['accuracy']) >= .99
    crossing = next((point['step'] for point in history if point['heldout']['accuracy'] >= .99), None)
    before = [point for point in history if crossing is None or point['step'] < crossing]
    eligible = [block for block in blocks(before, lambda point:
                point['train']['accuracy'] >= .99 and point['heldout']['accuracy'] <= .1)
                if len(block) >= 5 and block[-1]['step'] - block[0]['step'] >= 1_000]
    plateau = max(eligible, key=lambda block: (block[-1]['step'] - block[0]['step'], len(block)), default=None)
    memorized = None if plateau is None else {
        'start_step': plateau[0]['step'], 'end_step': plateau[-1]['step'],
        'observations': len(plateau), 'span_steps': plateau[-1]['step'] - plateau[0]['step'],
        'minimum_train_accuracy': min(point['train']['accuracy'] for point in plateau),
        'maximum_heldout_accuracy': max(point['heldout']['accuracy'] for point in plateau)}
    twenty = next((history[start:start + 20] for start in range(len(history) - 19)
                   if all(joint(point) for point in history[start:start + 20])), None)
    confirmation = None if twenty is None else {
        'onset_step': twenty[0]['step'], 'confirmed_step': twenty[-1]['step'],
        'observations': 20, 'span_steps': twenty[-1]['step'] - twenty[0]['step']}
    tail = [point for point in history if point['step'] >= budget - 50_000]
    if len(tail) != 50_000 // cadence + 1:
        raise ValueError('The final window is incomplete')
    failed = [point for point in tail if not joint(point)]
    worst = min(tail, key=lambda point: point['heldout']['accuracy'])
    final = {split: history[-1][split]['accuracy'] for split in ('train', 'heldout')}
    confirmed_by_plan = confirmation is not None and all(value == 1 for value in final.values())
    return {
        'canonical_observations': len(history), 'first_heldout_target_step': crossing,
        'memorized': memorized, 'confirmed_twenty_joint_evaluations': confirmation,
        'final_window': {
            'start_step': budget - 50_000, 'end_step': budget, 'observations': len(tail),
            'failed_observations': len(failed),
            'train_failed_observations': sum(point['train']['accuracy'] < .99 for point in tail),
            'heldout_failed_observations': sum(point['heldout']['accuracy'] < .99 for point in tail),
            'worst_heldout_accuracy': worst['heldout']['accuracy'], 'worst_heldout_step': worst['step'],
            'failures': [{'step': point['step'], 'train_accuracy': point['train']['accuracy'],
                          'heldout_accuracy': point['heldout']['accuracy']} for point in failed]},
        'final_accuracy': final, 'confirmed_by_step5_criterion': confirmed_by_plan,
        'persistent_final_performance': not failed,
        'stable_grokking_by_archived_criterion': bool(memorized and confirmation and not failed),
        'scope': 'Complete scheduled finite observations; no guarantee between evaluations or past the budget',
    }

use super::*;
pub fn check(input: &Request, out: &Report) -> Result<()> {
    let bound = input.validate()?;
    if out.input != *input
        || out.buffer_preflight_bytes != bound
        || out.stages.len() > input.policy.stages.saturating_sub(1)
        || out.states.len() > (1 + input.policy.stages) * input.policy.enumeration.points
    {
        return Err("component original input, retained-tree extent or bound changed".into());
    }
    if input.policy.stages == 0 {
        if out.base.is_some()
            || !out.stages.is_empty()
            || !out.states.is_empty()
            || !out.complete.is_empty()
            || out.reason != "stage_cap"
            || out.work != Work::default()
        {
            return Err("zero-stage component supplied states".into());
        }
        return Ok(());
    }
    let base = out
        .base
        .as_ref()
        .ok_or("missing original base enumeration")?;
    affine::check(&input.base_request(0), base)?;
    let mut work = Work::default();
    work.charge(base);
    let mut origins: Vec<_> = (0..base.points.len())
        .map(|point| Origin::Base { point })
        .collect();
    for (index, stage) in out.stages.iter().enumerate() {
        if stage.parent >= origins.len()
            || stage.parent >= out.states.len()
            || work.finite_scalar_updates >= input.policy.total_scalar_updates
        {
            return Err("component stage parent/order or remaining scalar budget differs".into());
        }
        let parent = &out.states[stage.parent];
        if parent.decision != (Decision::Expanded { stage: index })
            || parent.modulus >= input.factor.modulus
        {
            return Err("component expanded state is not its actual stage parent".into());
        }
        let request = input.lift_request(parent, work.finite_scalar_updates)?;
        lifts::check(&request, &stage.report)?;
        work.charge(&stage.report.digits);
        work.lift_original_products +=
            stage.report.work.base_products + stage.report.work.lifted_products;
        origins.extend((0..stage.report.branches.len()).map(|point| Origin::Digit {
            stage: index,
            point,
        }));
    }
    if origins.len() != out.states.len() {
        return Err("component skipped or added an enumerated branch".into());
    }
    let mut complete = vec![];
    for (index, (state, origin)) in out.states.iter().zip(&origins).enumerate() {
        let (raw, modulus, anchor, inverse) = match *origin {
            Origin::Base { point } => {
                let raw = &base.points[point].values;
                let anchor = (0..input.classes).find(|&c| raw[2 * input.n + c] != 0);
                let inverse = match anchor {
                    None => None,
                    Some(c) => {
                        let inv = state
                            .inverse
                            .ok_or("missing actual class normalization inverse")?;
                        if inv == 0
                            || inv >= input.factor.prime
                            || (inv as i128 * raw[2 * input.n + c] as i128)
                                % input.factor.prime as i128
                                != 1
                        {
                            return Err("component normalization divides by a nonunit".into());
                        }
                        work.inverse_tests += inv as usize;
                        work.normalization_products += input.system.variables;
                        Some(inv)
                    }
                };
                (raw, input.factor.prime, anchor, inverse)
            }
            Origin::Digit { stage, point } => {
                let lifted = &out.stages[stage];
                (
                    &lifted.report.branches[point].point,
                    lifted.report.input.modulus * lifted.report.input.prime,
                    out.states[lifted.parent].anchor,
                    None,
                )
            }
        };
        if state.origin != *origin
            || state.modulus != modulus
            || state.anchor != anchor
            || state.inverse != inverse
            || state.point.len() != input.system.variables
        {
            return Err("component branch source, anchor or shape differs".into());
        }
        let point: Vec<_> = raw
            .iter()
            .map(|&x| (x as i128 * inverse.unwrap_or(1) as i128).rem_euclid(modulus as i128) as u16)
            .collect();
        let mut counts = vec![0; modulus as usize];
        for &z in &point[2 * input.n..] {
            counts[z as usize] += 1;
            work.class_coordinates += 1;
        }
        if state.point != point || state.counts != counts {
            return Err("actual normalized point or class multiplicities differ".into());
        }
        let expected = input.classes / modulus as usize;
        let earlier = &out.states[..index];
        let decision = if anchor.is_none() {
            Decision::NoClassUnit
        } else if let Some(residue) = counts.iter().position(|&n| n != expected) {
            Decision::Multiplicity {
                residue,
                count: counts[residue],
                expected,
            }
        } else if let Some(state) = earlier.iter().position(|s| {
            s.decision.retained() && s.modulus == modulus && s.anchor == anchor && s.point == point
        }) {
            Decision::Duplicate { state }
        } else if earlier.iter().filter(|s| s.decision.retained()).count() == input.policy.states {
            Decision::StateCap
        } else {
            // Direct signed arithmetic on every original coefficient,
            // independent of the producer's normalization/point generation.
            for row in &input.system.rows {
                let dot: i128 = row
                    .iter()
                    .zip(&point)
                    .map(|(&a, &x)| a as i128 * x as i128)
                    .sum();
                if dot.rem_euclid(modulus as i128) != 0 {
                    return Err("retained component fails actual original equations".into());
                }
            }
            work.endpoint_products += (input.system.variables * input.system.rows.len()) as u64;
            if modulus == input.factor.modulus {
                complete.push(index);
                Decision::Complete
            } else if let Some(stage) = out.stages.iter().position(|s| s.parent == index) {
                Decision::Expanded { stage }
            } else {
                if work.stages == input.policy.stages {
                    Decision::StageCap
                } else if work.finite_scalar_updates == input.policy.total_scalar_updates {
                    Decision::ScalarCap
                } else {
                    return Err("component omitted an eligible uncapped expansion".into());
                }
            }
        };
        if state.decision != decision {
            return Err("component branch filter, duplicate or cap outcome differs".into());
        }
    }
    let reason = if complete.is_empty() {
        "no_complete_component"
    } else {
        "complete_component_candidates"
    };
    if out.complete != complete
        || out.reason != reason
        || out.work != work
        || work.finite_scalar_updates > input.policy.total_scalar_updates
    {
        return Err("component completed bank, actual work or termination differs".into());
    }
    schedule(input, out, base.points.len())?;
    Ok(())
}

// Independently reconstruct the queue from validated origins and retention
// decisions, including capped states. The producer does not supply a queue.
fn schedule(input: &Request, out: &Report, base_points: usize) -> Result<()> {
    let eligible = |s: &State| s.decision.retained() && s.modulus < input.factor.modulus;
    let mut queues: Vec<Vec<usize>> = out.states[..base_points]
        .iter()
        .enumerate()
        .filter(|(_, s)| eligible(s))
        .map(|(i, _)| vec![i])
        .collect();
    let mut next = 0;
    for (index, stage) in out.stages.iter().enumerate() {
        let active: Vec<_> = (0..queues.len())
            .filter(|&root| !queues[root].is_empty())
            .collect();
        let root = active
            .iter()
            .copied()
            .find(|&root| root >= next)
            .or_else(|| active.first().copied())
            .ok_or("component expanded an exhausted root bank")?;
        if queues[root].pop() != Some(stage.parent) {
            return Err("component round-robin/depth-first order differs".into());
        }
        for (state_index, state) in out.states.iter().enumerate().rev() {
            if matches!(state.origin, Origin::Digit { stage, .. } if stage == index)
                && eligible(state)
            {
                queues[root].push(state_index);
            }
        }
        next = root + 1;
    }
    if queues.iter().any(|queue| !queue.is_empty())
        && out.work.stages != input.policy.stages
        && out.work.finite_scalar_updates != input.policy.total_scalar_updates
    {
        return Err("component stopped with uncapped scheduled branches".into());
    }
    Ok(())
}

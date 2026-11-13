use super::*;
fn append(
    out: &mut Report,
    origin: Origin,
    modulus: u16,
    raw: &[u16],
    inherited: Option<usize>,
) -> Result<()> {
    let r = &out.input;
    let is_base = matches!(origin, Origin::Base { .. });
    let anchor = if is_base {
        (0..r.classes).find(|&c| raw[2 * r.n + c] != 0)
    } else {
        inherited
    };
    let inverse = if is_base {
        anchor.map(|c| {
            (1..modulus)
                .find(|&v| {
                    out.work.inverse_tests += 1;
                    v as u32 * raw[2 * r.n + c] as u32 % modulus as u32 == 1
                })
                .expect("checked nonzero prime-field coordinate")
        })
    } else {
        None
    };
    let point = if let Some(inverse) = inverse {
        out.work.normalization_products += r.system.variables;
        raw.iter()
            .map(|&x| (x as u32 * inverse as u32 % modulus as u32) as u16)
            .collect()
    } else {
        raw.to_vec()
    };
    let mut counts = vec![0; modulus as usize];
    for &c in &point[2 * r.n..] {
        counts[c as usize] += 1;
        out.work.class_coordinates += 1;
    }
    let expected = r.classes / modulus as usize;
    let decision = if anchor.is_none() {
        Decision::NoClassUnit
    } else if let Some(residue) = counts.iter().position(|&n| n != expected) {
        Decision::Multiplicity {
            residue,
            count: counts[residue],
            expected,
        }
    } else if let Some(state) = out.states.iter().position(|s| {
        s.decision.retained() && s.modulus == modulus && s.point == point && s.anchor == anchor
    }) {
        Decision::Duplicate { state }
    } else if out.states.iter().filter(|s| s.decision.retained()).count() == r.policy.states {
        Decision::StateCap
    } else {
        plan::original(r, modulus, &point)?;
        out.work.endpoint_products += (r.system.variables * r.system.rows.len()) as u64;
        if modulus == r.factor.modulus {
            Decision::Complete
        } else {
            Decision::Pending
        }
    };
    let index = out.states.len();
    if decision == Decision::Complete {
        out.complete.push(index);
    }
    out.states.push(State {
        origin,
        modulus,
        anchor,
        inverse,
        point,
        counts,
        decision,
    });
    Ok(())
}
pub fn run(input: Request) -> Result<Report> {
    let bound = input.validate()?;
    let mut out = Report {
        input,
        buffer_preflight_bytes: bound,
        base: None,
        stages: vec![],
        states: vec![],
        complete: vec![],
        reason: String::new(),
        work: Work::default(),
    };
    if out.input.policy.stages == 0 {
        out.reason = "stage_cap".into();
        return Ok(out);
    }
    let base = affine::run(out.input.base_request(0))?;
    out.work.charge(&base);
    for (index, point) in base.points.iter().enumerate() {
        append(
            &mut out,
            Origin::Base { point: index },
            base.input.problem.system.prime,
            &point.values,
            None,
        )?;
    }
    out.base = Some(base);
    // Round-robin across normalized base points, depth-first within each.
    // Wide branches that fail only at later powers cannot consume every
    // stage before another base point reaches its final modulus.
    let mut roots: Vec<_> = out
        .states
        .iter()
        .enumerate()
        .filter(|(_, s)| s.decision == Decision::Pending)
        .map(|(index, _)| vec![index])
        .collect();
    let mut cursor = 0;
    while let Some(root) = (0..roots.len())
        .map(|offset| (cursor + offset) % roots.len())
        .find(|&root| !roots[root].is_empty())
    {
        let cap = if out.work.stages == out.input.policy.stages {
            Some(Decision::StageCap)
        } else if out.work.finite_scalar_updates == out.input.policy.total_scalar_updates {
            Some(Decision::ScalarCap)
        } else {
            None
        };
        if let Some(cap) = cap {
            for state in &mut out.states {
                if state.decision == Decision::Pending {
                    state.decision = cap.clone();
                }
            }
            break;
        }
        let index = roots[root].pop().expect("selected nonempty root");
        cursor = (root + 1) % roots.len();
        let request = out
            .input
            .lift_request(&out.states[index], out.work.finite_scalar_updates)?;
        let lifted = lifts::run(request)?;
        out.work.charge(&lifted.digits);
        out.work.lift_original_products += lifted.work.base_products + lifted.work.lifted_products;
        let stage = out.stages.len();
        out.states[index].decision = Decision::Expanded { stage };
        let modulus = lifted.input.modulus * lifted.input.prime;
        let anchor = out.states[index].anchor;
        let start = out.states.len();
        for (point, branch) in lifted.branches.iter().enumerate() {
            append(
                &mut out,
                Origin::Digit { stage, point },
                modulus,
                &branch.point,
                anchor,
            )?;
        }
        roots[root].extend(
            (start..out.states.len())
                .rev()
                .filter(|&i| out.states[i].decision == Decision::Pending),
        );
        out.stages.push(Stage {
            parent: index,
            report: lifted,
        });
    }
    out.reason = if out.complete.is_empty() {
        "no_complete_component"
    } else {
        "complete_component_candidates"
    }
    .into();
    Ok(out)
}

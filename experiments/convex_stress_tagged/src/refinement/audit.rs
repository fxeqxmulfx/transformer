//! Independent raw-coordinate, integer-minimum and work-ledger checks.
//! All frequency spectra and ordinary CE are measured at the retained endpoint.
use super::*;
use serde_json::{Value, json};

pub fn check(input: &Input, out: &Outcome) -> Result<Value> {
    let samples = validate(input)?;
    let (n, q, v) = (input.seed.n, input.seed.q, input.seed.vocab);
    if &out.input != input || out.steps.is_empty() || out.steps.len() > 1 + 5 * input.rounds {
        return Err("free-phase audit input or trace length mismatch".into());
    }
    let mut classes = vec![];
    for o in &samples {
        if !classes.contains(&o.label) {
            classes.push(o.label);
        }
    }
    let (mut best, mut axis_checks, mut decoder_checks) = (None, 0, 0);
    let mut cost = Cost {
        class_observations: samples.len(),
        ..Cost::default()
    };
    for (i, step) in out.steps.iter().enumerate() {
        let s = &step.state;
        let model = s.model(v)?;
        if model.n != n || model.q != q {
            return Err("free-phase trace changed its dimensions".into());
        }
        let errors = samples
            .iter()
            .filter(|o| model.output_labels[o.cell] != o.label)
            .count();
        let mut counts = vec![0usize; q * v];
        for o in &samples {
            counts[model.latent_labels[o.cell] * v + o.label] += 1;
        }
        let missing: Vec<_> = counts
            .chunks(v)
            .enumerate()
            .filter(|(_, row)| row.iter().all(|&c| c == 0))
            .map(|(z, _)| z)
            .collect();
        if errors != s.errors || missing != s.unseen_latents {
            return Err("raw free-phase errors or missing-state coverage differs".into());
        }
        let expected = if i == 0 {
            Kind::Seed
        } else {
            [
                Kind::Decoder,
                Kind::Row,
                Kind::Decoder,
                Kind::Column,
                Kind::Decoder,
            ][(i - 1) % 5]
        };
        if step.kind != expected {
            return Err("free-phase update order mismatch".into());
        }
        cost.state_observations += samples.len();
        if i == 0 {
            if model != input.seed || step.minimum_costs.is_some() {
                return Err("free-phase seed changed".into());
            }
        } else {
            let old = &out.steps[i - 1].state;
            if errors > old.errors {
                return Err("raw observed CE increased".into());
            }
            match step.kind {
                Kind::Seed => unreachable!(),
                Kind::Decoder => {
                    decoder_checks += certificate::decoder(old, step, &counts, &classes, v)?;
                    cost.decoder_observations += samples.len();
                    cost.decoder_comparisons += q * classes.len();
                }
                Kind::Row | Kind::Column => {
                    axis_checks += certificate::axis(&samples, old, step, n, q)?;
                    cost.axis_steps += 1;
                    cost.axis_observation_comparisons += q * samples.len();
                    cost.axis_minimum_comparisons += n * q;
                }
            }
        }
        if missing.is_empty() && best.is_none_or(|b: usize| errors < out.steps[b].state.errors) {
            best = Some(i);
        }
    }
    if best != out.best || cost != out.cost || cost.axis_steps > input.axis_budget {
        return Err("free-phase endpoint or actual work ledger mismatch".into());
    }
    let endpoint = if let Some(i) = best {
        out.steps[i].state.model(v)?.audit(&samples)?
    } else {
        Value::Null
    };
    Ok(
        json!({"steps_checked":out.steps.len(),"state_observations":cost.state_observations,
        "axis_observation_comparisons":axis_checks,"axis_minimum_inequalities":cost.axis_minimum_comparisons,
        "decoder_count_comparisons":decoder_checks,"endpoint":endpoint,
        "integer_coordinate_minima":true,"full_latent_coverage_required":true,
        "joint_global_optimality_claimed":false,"exact_real_spectral_certificate":false}),
    )
}

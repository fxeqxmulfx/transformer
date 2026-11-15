//! Frozen synthetic/noise.py: one fixed uniform-incorrect label draw per
//! original prompt/output position, with structural EOS kept unchanged.
use crate::*;
use serde_json::{Value, json};

pub fn corrupt(e: &data::Example, rate: f64, seed: u32) -> Result<(data::Example, bool)> {
    if !rate.is_finite()
        || !(0. ..=1.).contains(&rate)
        || e.task != "parity"
        || e.answer.len() != 2
        || ![24, 25].contains(&e.answer[0])
        || e.answer[1] != 17
        || e.prompt.last() != Some(&18)
    {
        return Err("invalid bounded parity corruption".into());
    }
    let mut answer = e.answer.clone();
    let mut changed = false;
    if rate > 0. {
        // Source address uses answer position0, not its absolute tensor slot.
        let identity = json!([seed, "parity", [e.prompt, 0], [24, 25]]);
        let mut rng = wire::addressed_rng(&identity);
        if rng.random() < rate {
            let mut index = rng.below(1) as usize;
            index += usize::from(index >= usize::from(answer[0] == 25));
            answer[0] = [24, 25][index];
            changed = true;
        }
    }
    Ok((
        wire::answered("parity", e.prompt.clone(), answer, e.limit),
        changed,
    ))
}
pub fn apply(clean: &[data::Example], rate: f64, seed: u32) -> Result<(Vec<data::Example>, Value)> {
    if clean.is_empty() {
        return Err("empty noise pool".into());
    }
    let mut rows = Vec::with_capacity(clean.len());
    let mut changed = 0;
    for e in clean {
        let (row, flip) = corrupt(e, rate, seed)?;
        changed += usize::from(flip);
        rows.push(row);
    }
    Ok((
        rows,
        json!({"kernel":"uniform_incorrect_label","rate":rate,"seed":seed,
        "eligible_targets":clean.len(),"changed_targets":changed,
        "realized_rate":changed as f64/clean.len() as f64,
        "assignment":"fixed_causal_input_or_prompt_position"}),
    ))
}

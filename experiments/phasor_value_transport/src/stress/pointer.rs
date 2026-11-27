//! Check the attainable labels of the causal position-based Argand readout.
use super::{Example, Result, Value, cases, data, json};

fn generation_reachable(example: &Example) -> bool {
    // The frozen rollout stops at token 17 or at the example's limit.
    example.answer.len() <= example.limit
        && example.answer.split_last().is_some_and(|(_, prefix)| {
            !prefix.contains(&17)
                && example
                    .answer
                    .iter()
                    .all(|token| example.prompt.contains(token))
        })
}

fn split(examples: &[Example], vocab: usize) -> Value {
    let mut supervised = 0;
    let mut absent_targets = 0;
    let mut infinite_loss_examples = 0;
    let mut nonconstant_inputs = 0;
    let mut possibly_exact = 0;
    for example in examples {
        let mut finite = true;
        // The first token different from the opening token is a mandatory
        // unseen target for the all-token loss, regardless of any continuation.
        nonconstant_inputs += usize::from(example.tokens.windows(2).any(|pair| pair[0] != pair[1]));
        for (pos, &target) in example.targets.iter().enumerate() {
            if target < 0 {
                continue;
            }
            supervised += 1;
            if !example.tokens[..=pos].contains(&target) {
                absent_targets += 1;
                finite = false;
            }
        }
        infinite_loss_examples += usize::from(!finite);
        let reachable = if !example.prompt.is_empty() {
            generation_reachable(example)
        } else if let Some((base, future)) = example.field {
            // The frozen field scorer pools the probability of token aliases.
            example
                .tokens
                .iter()
                .any(|&token| (token - vocab as i64 / 2 + base).rem_euclid(97) == future)
        } else {
            finite
        };
        possibly_exact += usize::from(reachable);
    }
    json!({
        "examples": examples.len(),
        "supervised_positions": supervised,
        "absent_targets": absent_targets,
        "infinite_loss_examples": infinite_loss_examples,
        "nonconstant_inputs": nonconstant_inputs,
        "possibly_exact": possibly_exact,
        "exact_accuracy_upper_bound": possibly_exact as f64 / examples.len() as f64,
    })
}

pub fn audit() -> Result<()> {
    let mut tasks = Vec::new();
    for row in cases()? {
        let task = data::build(&row)?;
        let train = split(&task.splits["train"], task.vocab);
        let heldout = split(&task.splits[&task.selection], task.vocab);
        tasks.push(json!({
            "case": row["name"],
            "tier": row["tier"],
            "heldout_split": task.selection,
            "train": train,
            "heldout": heldout,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"tasks": tasks}))?
    );
    Ok(())
}

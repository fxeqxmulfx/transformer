//! Compare the causal successor address itself with supervised token values.
//! The walk uses input tokens alone. Labels select the scored positions.
use super::*;

pub fn run(name: &str, hops: usize) -> Result<()> {
    if hops == 0 {
        return Err("the successor walk must take at least one step".into());
    }
    let (_, task) = load(name)?;
    let mut splits = BTreeMap::new();
    for (split, examples) in task.splits {
        let (mut positions, mut written, mut hits, mut sequences, mut exact) = (0, 0, 0, 0, 0);
        for example in &examples {
            if !example.prompt.is_empty() || example.tokens.len() != example.targets.len() {
                return Err(
                    "successor alignment requires position-labelled input sequences".into(),
                );
            }
            let mut memory = BTreeMap::new();
            let mut labelled = 0;
            let mut correct = 0;
            for (i, (&token, &target)) in example.tokens.iter().zip(&example.targets).enumerate() {
                if i >= 2 {
                    memory.insert(example.tokens[i - 2], example.tokens[i - 1]);
                }
                if target >= 0 {
                    let mut value = token;
                    let mut complete = true;
                    for _ in 0..hops {
                        let Some(&next) = memory.get(&value) else {
                            complete = false;
                            break;
                        };
                        value = next;
                    }
                    labelled += 1;
                    written += usize::from(complete);
                    correct += usize::from(value == target);
                }
            }
            if labelled > 0 {
                positions += labelled;
                hits += correct;
                sequences += 1;
                exact += usize::from(correct == labelled);
            }
        }
        splits.insert(
            split,
            json!({"labelled_positions":positions,"written":written,"address_matches":hits,
            "supervised_sequences":sequences,"all_addresses_match":exact}),
        );
    }
    println!(
        "{}",
        json!({"case":name,"hops":hops,"selection":task.selection,"splits":splits})
    );
    Ok(())
}

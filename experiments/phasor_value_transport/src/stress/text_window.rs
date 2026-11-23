//! DiffGPT 5a37c515 probes.text. TextWindowScore proves chronological
//! separation and the token-weighted negative mean CE selection rule.
use basis::{data::Task, wire};
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

const CORPUS: &[u8] = include_bytes!("../../../../.lake/reproduction/gpt/data/tinyshakespeare.txt");
const CORPUS_SHA256: &str = "53493bf304b639aba6a47400da75c58ce32372b378fd2cd9d16ee987aeb12e4e";
const DIGESTS: &str = include_str!("../../tests/reference/text_window_sha256.tsv");
const CONTEXT: usize = 128;

pub(super) fn build(case: &Value) -> basis::Result<(Task, Value)> {
    if case["name"] != "tiny-shakespeare-gpt"
        || case["data_seed"] != 1
        || case["kind"] != "text"
        || case["metric"] != "negative_loss"
        || !CORPUS.is_ascii()
        || basis::hash(CORPUS) != CORPUS_SHA256
    {
        return Err("Tiny Shakespeare corpus or source protocol changed".into());
    }
    let alphabet: BTreeSet<_> = CORPUS.iter().copied().collect();
    let mut lookup = [0i64; 128];
    for (i, &symbol) in alphabet.iter().enumerate() {
        lookup[usize::from(symbol)] = i as i64;
    }
    let tokens: Vec<_> = CORPUS.iter().map(|&c| lookup[usize::from(c)]).collect();
    let first = tokens.len() * 8 / 10;
    let second = tokens.len() * 9 / 10;
    let mut splits = BTreeMap::new();
    let mut fingerprints = json!({});
    let pins: BTreeMap<_, _> = DIGESTS
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .collect();
    for (name, sequence, count, seed) in [
        ("train", &tokens[..first], 50_000, 1),
        ("validation", &tokens[first..second], 1024, 1001),
        ("test", &tokens[second..], 1024, 2001),
    ] {
        // CPU torch.randperm uses one forward Fisher–Yates swap per prefix entry.
        let mut starts: Vec<_> = (0..sequence.len() - CONTEXT).collect();
        let mut rng = basis::Random::torch_seed(seed);
        let mut rows = Vec::with_capacity(count);
        for i in 0..count {
            let j = i + rng.next_u32() as usize % (starts.len() - i);
            starts.swap(i, j);
            let chars = &sequence[starts[i]..][..=CONTEXT];
            let mut row = wire::example("text", chars[..CONTEXT].to_vec(), chars[1..].to_vec());
            row.changes.fill(false);
            rows.push(row);
        }
        let hash = wire::tensor_hash(&rows);
        if pins.get(name).copied() != Some(hash.as_str()) {
            return Err(
                format!("Tiny Shakespeare {name} ATen tensor fingerprint differs: {hash}").into(),
            );
        }
        fingerprints[name] = json!(hash);
        splits.insert(name.to_owned(), rows);
    }
    Ok((
        Task {
            splits,
            vocab: alphabet.len(),
            selection: "validation".into(),
        },
        fingerprints,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_windows_match_native_aten_on_every_split() -> basis::Result<()> {
        let (task, _) = build(
            &json!({"kind":"text","name":"tiny-shakespeare-gpt","metric":"negative_loss","data_seed":1}),
        )?;
        assert_eq!((task.vocab, task.selection.as_str()), (65, "validation"));
        for (split, count) in [("train", 50_000), ("validation", 1024), ("test", 1024)] {
            assert_eq!(task.splits[split].len(), count);
            assert!(task.splits[split].iter().all(|r| r.tokens.len() == CONTEXT
                && r.targets.len() == CONTEXT
                && r.tokens[1..] == r.targets[..CONTEXT - 1]
                && r.changes.iter().all(|&c| !c)));
        }
        Ok(())
    }
}

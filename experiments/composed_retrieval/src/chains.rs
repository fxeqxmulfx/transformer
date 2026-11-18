//! Apply the existing CM117 training protocol to the twelve original
//! final-answer chains. The model is constructed from object dimensions.
use crate::{
    Result,
    chain_data::{self, PRIMES, STEPS},
    data::{self, CaseSpec, Protocol},
};
use baseline::{data::Task, hash};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

pub const NAME: &str = "composed-retrieval-original-chains13-product-radii-v1";
pub const OVERLAP: &str = "arithmetic-add";

pub fn configure(p: &mut Protocol) -> Result<()> {
    let d: Value = serde_json::from_str(data::CHAIN_PROTOCOL)?;
    if d["name"] != NAME
        || d["overlap"] != OVERLAP
        || p.cases.len() != 64
        || !p.learned_product_radii
        || p.signed_profile
        || p.data_seed != 1
        || d["data_seed"] != p.data_seed
        || d["training_protocol_sha256"] != hash(data::RADII_PROTOCOL.as_bytes())
        || d["chain_source_manifest_sha256"] != hash(chain_data::SOURCES.as_bytes())
        || d["cases"].as_object().is_none_or(|m| m.len() != 12)
        || d["batch"] != 256
        || d["selection"] != "validation"
        || d["metric"] != "sequence_accuracy"
        || d["target"] != 0.99
    {
        return Err("changed original chain protocol or CM117 parent".into());
    }
    let anchor = p
        .cases
        .get(OVERLAP)
        .ok_or("missing parent addition case")?
        .clone();
    let mut cases = BTreeMap::from([(OVERLAP.into(), anchor)]);
    for prime in PRIMES {
        for steps in STEPS {
            let id = format!("modular-chain-p{prime}-l{steps}");
            let pin = d["cases"][&id]
                .as_str()
                .ok_or("original chain tensor fingerprints")?;
            if p.cases.contains_key(&id)
                || pin.len() != 64
                || !pin.bytes().all(|x| x.is_ascii_hexdigit())
            {
                return Err("unexpected chain overlap or digest".into());
            }
            cases.insert(
                id,
                CaseSpec {
                    vocabulary: prime,
                    context: 1 + 4 * steps,
                    batch: 256,
                    train: 2048,
                    selection: "validation".into(),
                    metric: "sequence_accuracy".into(),
                    target: 0.99,
                    metadata_sha256: Some(pin.into()),
                    source_fingerprints: None,
                },
            );
        }
    }
    p.cases = cases;
    p.name = NAME.into();
    Ok(())
}

pub fn original(s: &CaseSpec, seed: u32) -> Result<(Task, Value)> {
    let steps = s
        .context
        .checked_sub(1)
        .ok_or("missing original chain context")?
        / 4;
    let task = chain_data::build(s.vocabulary, steps, u64::from(seed))?;
    let mut pins = BTreeMap::new();
    for (name, rows) in &task.splits {
        pins.insert(
            name,
            chain_data::fingerprints(rows, task.vocab)?.dense_tensor_sha256,
        );
    }
    let pins = json!(pins);
    if s.context != 1 + 4 * steps
        || s.train != task.splits["train"].len()
        || s.selection != task.selection
        || s.metadata_sha256.as_deref() != Some(&hash(&serde_json::to_vec(&pins)?))
    {
        return Err("original chain records, masks or source dimensions differ".into());
    }
    Ok((task, pins))
}

pub fn inventory(p: &Protocol, inputs: &Path) -> Result<Value> {
    let source = chain_data::inventory::inventory(inputs)?;
    let mut entries = BTreeMap::new();
    let mut one = 0u64;
    for (id, s) in &p.cases {
        let (task, pins) = data::original(p, id)?;
        let mut splits = BTreeMap::new();
        for (name, rows) in &task.splits {
            if id != OVERLAP
                && (source["cases"][id][name]["dense_tensor_sha256"] != pins[name]
                    || source["cases"][id][name]["examples"] != rows.len())
            {
                return Err("archived source inventory differs from the optimizer records".into());
            }
            let count = rows
                .iter()
                .flat_map(|r| &r.targets)
                .filter(|&&x| x >= 0)
                .count();
            if count != rows.len() {
                return Err("expected one supervised object per row".into());
            }
            one += 8 * s.vocabulary as u64 * count as u64;
            splits.insert(name, json!({"examples":rows.len(),"targets":count}));
        }
        entries.insert(id,json!({"dimensions":{"vocabulary":s.vocabulary,"splits":splits},"split_fingerprints":pins}));
    }
    let cells = p.seeds.len() * p.modes.len();
    let bytes = one * 4 * cells as u64;
    if entries.len() != 13 || cells != 6 || bytes != 1_140_971_712 {
        return Err("incomplete original chain or control forecast".into());
    }
    Ok(
        json!({"protocol_sha256":hash(data::CHAIN_PROTOCOL.as_bytes()),"cases":entries,"fits":2*cells*p.cases.len(),
        "raw_bytes":bytes,"chain_source":source}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{model::Model, train};

    #[test]
    fn both_optimizers_take_a_complete_original_final_answer_minibatch() -> Result<()> {
        let mut p = data::parent_protocol()?;
        configure(&mut p)?;
        p.updates = 1;
        let s = &p.cases["modular-chain-p17-l1"];
        let (task, _) = original(s, p.data_seed)?;
        let rows = &task.splits["train"];
        let start = Model::new(task.vocab, s.context, &p, 0)?;
        for (row, sample) in rows.iter().zip(train::samples(&start, rows)?) {
            assert_eq!(sample.len(), 1);
            assert!(
                sample[0]
                    .prefix
                    .iter()
                    .copied()
                    .eq(row.tokens.iter().map(|&x| x as usize))
            );
            assert_eq!(sample[0].target, row.targets[2] as usize);
        }
        let mut batches = Vec::new();
        for mode in &p.modes {
            let mut model = start.clone();
            let receipt = train::fit(&mut model, rows, s.batch, mode, 0, &p, || Ok(()))?;
            assert_eq!(receipt["examples"], 256);
            assert_eq!(receipt["targets"], 256);
            assert_eq!(receipt["min_visits"], 1);
            assert_eq!(
                receipt["forward_loss_queries"],
                if mode == "forward" {
                    2 * model.weights.len()
                } else {
                    0
                }
            );
            assert_ne!(model.weights, start.weights);
            batches.push(receipt["batch_sha256"].clone());
        }
        assert_eq!(batches[0], batches[1]);
        Ok(())
    }

    #[test]
    fn source_cases_retain_cm117_coordinates_batches_and_one_visible_final_target() -> Result<()> {
        let parent = data::parent_protocol()?;
        let mut p = parent.clone();
        configure(&mut p)?;
        assert!(p.cases[OVERLAP] == parent.cases[OVERLAP]);
        assert!(p.learned_product_radii && !p.signed_profile);
        assert_eq!(p.updates, 1024);
        let s = &p.cases["modular-chain-p17-l1"];
        let (task, _) = original(s, p.data_seed)?;
        assert_eq!(
            (s.vocabulary, s.train, s.batch, s.context),
            (17, 2048, 256, 5)
        );
        let mut wrong = s.clone();
        wrong.context = 9;
        assert!(original(&wrong, p.data_seed).is_err());
        assert!(original(s, 11).is_err());
        assert!(
            task.splits["train"]
                .iter()
                .all(|e| e.tokens.len() == 3 && e.targets[..2] == [-100, -100])
        );
        Ok(())
    }
}

//! Original metrics and exposure diagnostics run only on sealed endpoints.
//! All counts below are calculated after prediction from complete raw logits.
use crate::{Result, cohort::Run, data, model::Model, operator::Tables};
use baseline::{
    channel::argmax,
    data::Example,
    metrics::{self, Predictor},
};
use bilinear::{
    experiment::{hash, records, write_json},
    train::Record,
};
use learned_retrieval::{
    binding_counts::{Context, Counts, Observation},
    cohort::{publish, raw_file, read},
};
use serde_json::{Value, json};
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

type Pairs = BTreeSet<(i64, usize)>;
struct Exposure {
    input: Vec<usize>,
    positive: Vec<usize>,
    queried: Pairs,
    stored: Pairs,
    binding: bool,
}
impl Exposure {
    fn new(tasks: &[Example], v: usize, binding: bool) -> Result<Self> {
        let mut e = Self {
            input: vec![0; v],
            positive: vec![0; v],
            queried: Pairs::new(),
            stored: Pairs::new(),
            binding,
        };
        for task in tasks {
            for &id in &task.tokens {
                *e.input
                    .get_mut(usize::try_from(id)?)
                    .ok_or("invalid input")? += 1;
            }
        }
        for r in records(tasks, &(0..tasks.len()).collect::<Vec<_>>())? {
            *e.positive
                .get_mut(r.target)
                .ok_or("invalid positive target")? += 1;
            if binding {
                let ctx = Context::new(&r)?;
                e.queried
                    .insert((*r.prefix.last().ok_or("missing query")?, r.target));
                e.stored
                    .extend([r.prefix[1], r.prefix[3]].into_iter().zip(ctx.values));
            }
        }
        Ok(e)
    }
    fn report(&self, rows: &[Record<'_>], bytes: &[u8], v: usize) -> Result<Value> {
        let mut groups = BTreeMap::from([
            ("seen_positive", [0, 0]),
            ("input_only", [0, 0]),
            ("absent", [0, 0]),
        ]);
        let mut unseen_queried = [0, 0];
        let mut unseen_stored = [0, 0];
        let mut paired = Counts::default();
        for (r, raw) in rows.iter().zip(bytes.chunks_exact(8 * v)) {
            let z: Vec<_> = raw
                .as_chunks::<8>()
                .0
                .iter()
                .map(|&b| f64::from_le_bytes(b))
                .collect();
            let hit = usize::from(argmax(&z) == r.target);
            let key = if self.positive[r.target] > 0 {
                "seen_positive"
            } else if self.input[r.target] > 0 {
                "input_only"
            } else {
                "absent"
            };
            let counts = groups.get_mut(key).unwrap();
            counts[0] += hit;
            counts[1] += 1;
            if self.binding {
                let ctx = Context::new(r)?;
                paired.observe(&Observation::new(&ctx, &z)?);
                let pair = (*r.prefix.last().ok_or("query")?, r.target);
                if !self.queried.contains(&pair) {
                    unseen_queried[0] += hit;
                    unseen_queried[1] += 1;
                }
                if !self.stored.contains(&pair) {
                    unseen_stored[0] += hit;
                    unseen_stored[1] += 1;
                }
            }
        }
        Ok(
            json!({"correct_total_by_exposure":groups,"unseen_queried_pair":self.binding.then_some(unseen_queried),
            "unseen_stored_pair":self.binding.then_some(unseen_stored),"binding":self.binding.then(||paired.report())}),
        )
    }
}
struct Raw<'a> {
    model: &'a Model,
    table: Tables,
    bytes: RefCell<Vec<u8>>,
}
impl Predictor for Raw<'_> {
    fn vocab(&self) -> usize {
        self.model.vocabulary
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        let ids = self
            .model
            .objects(prefix)
            .expect("validated benchmark prefix");
        let z = self
            .table
            .logits(self.model, &self.table.pass(self.model, &ids).hidden);
        let mut bytes = self.bytes.borrow_mut();
        for v in &z {
            bytes.extend(v.to_le_bytes());
        }
        z
    }
}
fn split(m: &Model, examples: &[Example], exposure: &Exposure) -> Result<(Value, Vec<u8>)> {
    #[cfg(feature = "rewrite-cohort")]
    if examples.first().is_some_and(|e| !e.prompt.is_empty()) {
        return generation_split(m, examples, exposure);
    }
    if examples.iter().any(|e| {
        !e.prompt.is_empty()
            || e.field.is_some()
            || e.tokens.len() != e.changes.len()
            || !e.targets.iter().any(|&t| t >= 0)
    }) {
        return Err("expected aligned non-generative records".into());
    }
    let rows = records(examples, &(0..examples.len()).collect::<Vec<_>>())?;
    let mut lengths = BTreeSet::new();
    for r in &rows {
        m.objects(r.prefix)?;
        if r.target >= m.vocabulary {
            return Err("invalid label".into());
        }
        lengths.insert(r.prefix.len().saturating_sub(1));
    }
    let predictor = Raw {
        model: m,
        table: Tables::new(m, &lengths.into_iter().collect::<Vec<_>>())?,
        bytes: RefCell::new(vec![]),
    };
    let original = metrics::evaluate(&predictor, examples)?;
    let bytes = predictor.bytes.into_inner();
    if bytes.len() != rows.len() * m.vocabulary * 8 {
        return Err("original scorer did not emit one complete output per record".into());
    }
    let groups = exposure.report(&rows, &bytes, m.vocabulary)?;
    Ok((
        json!({"exposure":groups,"raw_sha256":hash(&bytes),"raw99":original["token_accuracy"].as_f64().unwrap()>=0.99,
        "exact99":original["sequence_accuracy"].as_f64().unwrap()>=0.99,"metrics":original}),
        bytes,
    ))
}

#[cfg(feature = "rewrite-cohort")]
fn generation_split(
    m: &Model,
    examples: &[Example],
    exposure: &Exposure,
) -> Result<(Value, Vec<u8>)> {
    use crate::generation;
    struct Traced<'a> {
        predictor: Raw<'a>,
        prefixes: RefCell<Vec<u8>>,
    }
    impl Predictor for Traced<'_> {
        fn vocab(&self) -> usize {
            self.predictor.vocab()
        }
        fn logits(&self, prefix: &[i64]) -> Vec<f64> {
            generation::record_prefix(&mut self.prefixes.borrow_mut(), prefix);
            self.predictor.logits(prefix)
        }
    }
    let mut lengths = BTreeSet::new();
    for e in examples {
        generation::validate_source(e, m.vocabulary)?;
        m.objects(&e.tokens)?;
        let first_memory = e.prompt.len() - 1;
        lengths.extend(first_memory..first_memory + e.limit);
    }
    let raw = Raw {
        model: m,
        table: Tables::new(m, &lengths.into_iter().collect::<Vec<_>>())?,
        bytes: RefCell::default(),
    };
    let predictor = Traced {
        predictor: raw,
        prefixes: RefCell::default(),
    };
    let original = metrics::evaluate(&predictor, examples)?;
    let bytes = predictor.predictor.bytes.into_inner();
    let prefix_hash = hash(&predictor.prefixes.into_inner());
    let teacher = generation::decode(examples, m.vocabulary, &bytes, &prefix_hash, &original)?;
    let rows = records(examples, &(0..examples.len()).collect::<Vec<_>>())?;
    let groups = exposure.report(&rows, &teacher, m.vocabulary)?;
    Ok((
        json!({"exposure":groups,"exposure_inputs":"teacher_forced","raw_sha256":hash(&bytes),
        "query_prefixes_sha256":prefix_hash,"teacher_logits_sha256":hash(&teacher),
        "raw99":original["token_accuracy"].as_f64().ok_or("generated accuracy")?>=0.99,
        "exact99":original["sequence_accuracy"].as_f64().ok_or("generated exact")?>=0.99,
        "metrics":original}),
        bytes,
    ))
}
pub fn score(run: &Run<'_>, id: &str, replay: bool) -> Result<()> {
    run.both_sealed()?;
    let p = &run.protocol;
    let (task, _) = data::original(p, id)?;
    let s = &p.cases[id];
    let root = run.root(run.arm).join(id);
    let exposure = Exposure::new(
        &task.splits["train"],
        task.vocab,
        id == "gpt-probe-key-value-binding",
    )?;
    let evaluate = |out: &Path| -> Result<()> {
        let mut report = BTreeMap::new();
        for mode in &p.modes {
            for &seed in &p.seeds {
                run.admit()?;
                let cell = format!("{mode}-{seed}");
                let final_model: Model =
                    serde_json::from_slice(&fs::read(root.join(&cell).join("model.json"))?)?;
                let initial = Model::new(task.vocab, s.context, p, seed)?;
                for (phase, m) in [("initial", &initial), ("final", &final_model)] {
                    let key = format!("{cell}-{phase}");
                    let mut splits = BTreeMap::new();
                    for (name, examples) in &task.splits {
                        let (value, bytes) = split(m, examples, &exposure)?;
                        let raw = raw_file(out, &key, name);
                        if replay {
                            if fs::read(&raw)? != bytes {
                                return Err(format!("replay mismatch {}", raw.display()).into());
                            }
                        } else {
                            fs::write(&raw, &bytes)?;
                        }
                        splits.insert(name, value);
                    }
                    let success = splits[&s.selection]["metrics"][&s.metric]
                        .as_f64()
                        .ok_or("missing original criterion")?
                        >= s.target;
                    let strict = splits
                        .iter()
                        .filter(|(k, _)| !k.starts_with("train"))
                        .all(|(_, v)| v["raw99"] == true && v["exact99"] == true);
                    report.insert(key,json!({"original_success":success,"strict_all_heldout":strict,"splits":splits}));
                }
            }
        }
        let value = json!({"case":id,"protocol_sha256":crate::cohort::protocol_hash(),"training_input_counts":exposure.input,
            "training_positive_counts":exposure.positive,"selection":s.selection,"metric":s.metric,"target":s.target,"cells":report});
        let path = out.join("report.json");
        if replay {
            if read(&path)? != value {
                return Err("raw replay metrics or exposure counts differ".into());
            }
            Ok(())
        } else {
            write_json(&path, &value)
        }
    };
    let dir = root.join("scores");
    if replay {
        evaluate(&dir)
    } else {
        publish(&dir, evaluate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "rewrite-cohort")]
    #[test]
    fn live_rewrite_scores_record_each_original_and_model_written_prefix() -> Result<()> {
        let p = data::protocol()?;
        let id = "suite-copy-repeat";
        let (task, _) = data::original(&p, id)?;
        let model = Model::new(task.vocab, p.cases[id].context, &p, 0)?;
        let rows = &task.splits["train"][..2];
        let exposure = Exposure::new(rows, task.vocab, false)?;
        let (scored, bytes) = split(&model, rows, &exposure)?;
        let m = &scored["metrics"];
        let targets: usize = rows.iter().map(|e| e.answer.len()).sum();
        assert_eq!(m["targets"], targets);
        let generated = m["generated_tokens"].as_u64().unwrap() as usize;
        assert_eq!(bytes.len(), (targets + generated) * task.vocab * 8);
        let teacher = crate::generation::decode(
            rows,
            task.vocab,
            &bytes,
            scored["query_prefixes_sha256"].as_str().unwrap(),
            m,
        )?;
        learned_retrieval_transfer::audit::decode_split(
            &teacher,
            rows,
            task.vocab,
            &rows.iter().flat_map(|e| e.answer.iter().copied()).collect(),
            &json!({"raw_logits_sha256":scored["teacher_logits_sha256"],"metrics":m["teacher_forced"]}),
        )?;
        Ok(())
    }

    #[test]
    fn original_metrics_keep_both_queries_and_count_absent_answers() -> Result<()> {
        let m = Model::new(5, 4, &data::protocol()?, 0)?;
        let t = Tables::new(&m, &[1, 2])?;
        let winner = |x: &[usize]| argmax(&t.logits(&m, &t.pass(&m, x).hidden));
        let (y, z) = (winner(&[1, 2]), winner(&[1, 2, 3]));
        let right = Example {
            task: "fixture".into(),
            tokens: vec![1, 2, 3],
            targets: vec![-100, y as i64, z as i64],
            changes: vec![false, true, true],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        };
        let mut wrong = right.clone();
        let missing = (0..m.vocabulary).find(|&c| c != y && c != z).unwrap();
        wrong.targets[2] = missing as i64;
        let exposure = Exposure::new(std::slice::from_ref(&right), m.vocabulary, false)?;
        let (out, raw) = split(&m, &[right, wrong], &exposure)?;
        assert_eq!(out["metrics"]["correct"], 3);
        assert_eq!(out["metrics"]["exact"], 1);
        assert_eq!(out["metrics"]["token_accuracy"], 0.75);
        assert_eq!(out["metrics"]["sequence_accuracy"], 0.5);
        assert_eq!(out["raw99"], false);
        assert_eq!(out["exact99"], false);
        assert_eq!(raw.len(), 4 * m.vocabulary * 8);
        assert_eq!(
            out["exposure"]["correct_total_by_exposure"]["seen_positive"],
            json!([3, 3])
        );
        let class = if exposure.input[missing] > 0 {
            "input_only"
        } else {
            "absent"
        };
        assert_eq!(
            out["exposure"]["correct_total_by_exposure"][class],
            json!([0, 1])
        );
        Ok(())
    }
}

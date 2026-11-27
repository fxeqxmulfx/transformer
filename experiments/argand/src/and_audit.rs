//! Exhaustive post-training checks of the two finite Boolean AND tasks. The
//! canonical novel split is small; these checks cover every other bit string
//! of the training lengths and distinguish strings outside generator support.
use crate::{
    Result,
    data::Example,
    engine::Engine,
    metrics,
    suite::{self, load_weights, publish},
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Default)]
struct Counts {
    total: usize,
    correct: usize,
    min_gap: Option<f64>,
}
impl Counts {
    fn add(&mut self, gap: f64) {
        self.total += 1;
        self.correct += usize::from(gap > 0.0);
        self.min_gap = Some(self.min_gap.map_or(gap, |v| v.min(gap)));
    }
    fn json(&self) -> Value {
        json!({"correct":self.correct,"total":self.total,"min_gap":self.min_gap,
            "all_strict":self.total>0 && self.correct==self.total})
    }
}

pub(crate) fn run(dir: &Path) -> Result<Value> {
    let r: Value = serde_json::from_slice(&fs::read(dir.join("result.json"))?)?;
    let name = r["identity"]["case"].as_str().ok_or("missing case")?;
    if !matches!(name, "suite-boolean-and-shift" | "suite-boolean-and-random") {
        return Err("expected a native Boolean AND checkpoint".into());
    }
    let (row, task) = suite::load(name)?;
    let shape = suite::shape(&row, &task)?;
    let batch = shape.max_batch;
    let mut engine = Engine::new(shape, 1, 0)?;
    load_weights(dir, "weights.f32", &r["weights_sha256"], &mut engine)?;
    let train: BTreeSet<_> = task.splits["train"]
        .iter()
        .map(|e| e.prompt.clone())
        .collect();
    let (mut seen, mut native, mut novel_one, mut multi) = (
        Counts::default(),
        Counts::default(),
        Counts::default(),
        Counts::default(),
    );
    let mut p = metrics::Packed::default();
    let shifted = name == "suite-boolean-and-shift";
    for length in 8..=16 {
        for first in (0..1_u32 << length).step_by(batch) {
            let stop = (first + batch as u32).min(1 << length);
            let mut inputs = Vec::with_capacity(batch);
            let mut kinds = Vec::with_capacity(batch);
            for zeros in first..stop {
                let mut prompt = vec![1];
                prompt.extend((0..length).map(|bit| if zeros & (1 << bit) != 0 { 21 } else { 22 }));
                prompt.push(18);
                let early = length - (length / 4).max(1);
                kinds.push((
                    train.contains(&prompt),
                    zeros.count_ones(),
                    !shifted || (zeros >> early) == 0,
                ));
                inputs.push(Example::generation(
                    "boolean_and",
                    prompt,
                    vec![if zeros == 0 { 16 } else { 15 }, 17],
                ));
            }
            p.fill(&inputs, &(0..inputs.len() as u32).collect::<Vec<_>>());
            for ((seen_prompt, zeros, region), margin) in kinds
                .into_iter()
                .zip(crate::margins::rows(&mut engine, &p)?)
            {
                let gap = margin.gap;
                if seen_prompt {
                    seen.add(gap);
                } else if zeros <= 1 {
                    novel_one.add(gap);
                    if region {
                        native.add(gap);
                    }
                } else {
                    multi.add(gap);
                }
            }
        }
    }
    if seen.total != train.len() || seen.total + novel_one.total + multi.total != 130816 {
        return Err("exhaustive AND domain size mismatch".into());
    }
    let report = json!({"case":name,"weights_sha256":r["weights_sha256"],"seen_distinct":seen.json(),
        "unseen_native_support":native.json(),"unseen_at_most_one_zero_anywhere":novel_one.json(),
        "two_or_more_zeros":multi.json(),"hidden_correct":novel_one.correct+multi.correct,
        "hidden_total":novel_one.total+multi.total,"all_hidden_strict":novel_one.correct+multi.correct==novel_one.total+multi.total});
    publish(
        dir,
        "and-exhaustive.json",
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(report)
}

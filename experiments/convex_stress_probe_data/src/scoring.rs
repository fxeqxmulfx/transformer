//! DiffGPT probes.py NumericMetrics and the original causal generation scorer.
//! MSE uses every unquantized component, with penalty64 for invalid/missing ones.
//! Float64 stable CE is a native measurement, not a bitwise CUDA-loss claim.
use crate::{Result, boundary::Predictor, dataset::Dataset, generation};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts {
    pub target_count: u64,
    pub correct: u64,
    pub example_count: u64,
    pub exact: u64,
    pub transition_count: u64,
    pub transition_correct: u64,
    /// Each entry retains original class multiplicity and correctly emitted tokens.
    pub classes: BTreeMap<i64, [u64; 2]>,
}
impl Counts {
    fn row(&mut self, targets: &[i64], changes: &[bool], prediction: &[usize]) {
        let mut exact = prediction.len() == targets.len();
        self.example_count += 1;
        for (i, (&label, &changed)) in targets.iter().zip(changes).enumerate() {
            let hit = prediction.get(i).copied() == Some(label as usize);
            let entry = self.classes.entry(label).or_default();
            entry[0] += 1;
            entry[1] += u64::from(hit);
            self.target_count += 1;
            self.correct += u64::from(hit);
            self.transition_count += u64::from(changed);
            self.transition_correct += u64::from(changed && hit);
            exact &= hit;
        }
        self.exact += u64::from(exact);
    }
    pub fn raw99(&self) -> bool {
        self.target_count > 0 && 100 * self.correct >= 99 * self.target_count
    }
    pub fn exact99(&self) -> bool {
        self.example_count > 0 && 100 * self.exact >= 99 * self.example_count
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Score {
    pub raw: Counts,
    pub teacher_forced: Counts,
    pub cross_entropy: f64,
    pub example_cross_entropy: f64,
    pub generated: bool,
    pub ended: u64,
    pub generated_tokens: u64,
    pub extra_tokens: u64,
    pub numeric_components: u64,
    pub valid_numeric_components: u64,
    pub squared_error: f64,
    pub mse: Option<f64>,
    pub raw99: bool,
    pub exact99: bool,
    pub original_success: bool,
}
fn logits(model: &mut impl Predictor, prefix: &[i64], vocab: usize) -> Result<(Vec<f64>, usize)> {
    let values = model.logits(prefix)?;
    if values.len() != vocab || values.iter().any(|x| !x.is_finite()) {
        return Err("predictor returned invalid class logits".into());
    }
    let mut best = 0;
    for i in 1..values.len() {
        if values[i] > values[best] {
            best = i;
        }
    }
    Ok((values, best))
}
pub fn score(dataset: &Dataset, split: &str, model: &mut impl Predictor) -> Result<Score> {
    let shard = dataset.splits.get(split).ok_or("unknown source split")?;
    if shard.records.is_empty() {
        return Err("empty evaluation split".into());
    }
    match (
        &shard.teacher_words,
        dataset.numeric_step,
        dataset.numeric_offset,
    ) {
        (Some(teachers), Some(step), Some(_))
            if teachers.len() == shard.records.len() && step.is_finite() && step > 0. => {}
        (None, None, None) => {}
        _ => return Err("numeric scoring metadata or complete teacher inventory differs".into()),
    }
    let generates = !split.starts_with("train") && !shard.records[0].prompt.is_empty();
    let mut out = Score {
        raw: Counts::default(),
        teacher_forced: Counts::default(),
        cross_entropy: 0.,
        example_cross_entropy: 0.,
        generated: generates,
        ended: 0,
        generated_tokens: 0,
        extra_tokens: 0,
        numeric_components: 0,
        valid_numeric_components: 0,
        squared_error: 0.,
        mse: None,
        raw99: false,
        exact99: false,
        original_success: false,
    };
    for (index, row) in shard.records.iter().enumerate() {
        row.validate(dataset.vocabulary)?;
        if generates != (!split.starts_with("train") && !row.prompt.is_empty()) {
            return Err("mixed original evaluation layouts".into());
        }
        let mut expected = vec![];
        let mut changes = vec![];
        let mut predicted = vec![];
        let mut loss = 0.;
        for (position, &target) in row.targets.iter().enumerate() {
            if target == -100 {
                continue;
            }
            let (values, best) = logits(model, &row.tokens[..=position], dataset.vocabulary)?;
            let max = values[best];
            let sum: f64 = values.iter().map(|x| (x - max).exp()).sum();
            loss += sum.ln() + (max - values[target as usize]);
            expected.push(target);
            changes.push(row.changes[position]);
            predicted.push(best);
        }
        out.cross_entropy += loss;
        out.example_cross_entropy += loss / expected.len() as f64;
        out.teacher_forced.row(&expected, &changes, &predicted);
        if generates {
            predicted = generation::rollout(
                &row.prompt,
                row.generation_limit,
                dataset.vocabulary,
                |prefix| Ok(logits(model, prefix, dataset.vocabulary)?.1),
            )?;
            let counts = generation::score(&row.answer, &predicted)?;
            out.ended += u64::from(counts.ended);
            out.generated_tokens += counts.generated_tokens as u64;
            out.extra_tokens += counts.extra_tokens as u64;
        }
        out.raw.row(&expected, &changes, &predicted);
        if let Some(teachers) = &shard.teacher_words {
            let teacher = teachers.get(index).ok_or("missing numeric teacher row")?;
            let step = dataset.numeric_step.ok_or("missing numeric step")?;
            let offset = dataset.numeric_offset.ok_or("missing numeric offset")?;
            if teacher.len() + 1 != row.answer.len() {
                return Err("numeric teacher width".into());
            }
            for (component, &word) in teacher.iter().enumerate() {
                let truth = f64::from_bits(word);
                if !truth.is_finite() {
                    return Err("nonfinite numeric teacher".into());
                }
                let token = predicted.get(component).copied();
                let valid = token.is_some_and(|x| x >= 36 && x < dataset.vocabulary);
                out.numeric_components += 1;
                out.valid_numeric_components += u64::from(valid);
                out.squared_error += if valid {
                    let decoded = (token.unwrap() as i64 - 36 - offset) as f64 * step;
                    (decoded - truth).powi(2)
                } else {
                    64.
                };
            }
        }
    }
    out.cross_entropy /= out.teacher_forced.target_count as f64;
    out.example_cross_entropy /= out.teacher_forced.example_count as f64;
    if !out.cross_entropy.is_finite() || !out.squared_error.is_finite() {
        return Err("nonfinite native score".into());
    }
    out.raw99 = out.raw.raw99();
    out.exact99 = out.raw.exact99();
    out.mse =
        (out.numeric_components > 0).then(|| out.squared_error / out.numeric_components as f64);
    out.original_success = out.mse.map_or(out.exact99, |mse| mse <= 0.001);
    Ok(out)
}

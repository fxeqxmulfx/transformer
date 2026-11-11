//! Metrics ported from diff-gpt 3555299, complex_geometry_suite/native/src/metrics.rs.
//! Same complete-split, greedy rollout, EOS and residue-pooling semantics; CPU f64
//! logits replace the source's CUDA f32 buffers. Only supervised prefixes are scored.
use crate::{
    Result,
    channel::{Channel, argmax, cross_entropy},
    data::Example,
};
use serde_json::{Value, json};

/// Evaluate live joint coefficients without materializing a large checkpoint at
/// every observation. The saved Channel implements the identical prediction API.
pub trait Predictor {
    fn vocab(&self) -> usize;
    fn logits(&self, prefix: &[i64]) -> Vec<f64>;
}
impl Predictor for Channel {
    fn vocab(&self) -> usize {
        self.adapter.vocab
    }
    fn logits(&self, prefix: &[i64]) -> Vec<f64> {
        Channel::logits(self, prefix)
    }
}

#[derive(Default)]
struct Counts {
    loss: f64,
    example_loss: f64,
    targets: usize,
    correct: usize,
    exact: usize,
    transitions: usize,
    transition_correct: usize,
    examples: usize,
    classes: Vec<usize>,
    right: Vec<usize>,
}
impl Counts {
    fn new(vocab: usize) -> Self {
        Self {
            classes: vec![0; vocab],
            right: vec![0; vocab],
            ..Self::default()
        }
    }
    fn observe(&mut self, label: usize, hit: bool, transition: bool) {
        self.targets += 1;
        self.correct += usize::from(hit);
        self.classes[label] += 1;
        self.right[label] += usize::from(hit);
        if transition {
            self.transitions += 1;
            self.transition_correct += usize::from(hit);
        }
    }
    fn report(&self) -> Value {
        let classes: serde_json::Map<String, Value> = self
            .classes
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(label, &count)| {
                (
                    label.to_string(),
                    json!({"name":label.to_string(),"count":count,
                "correct":self.right[label],"accuracy":self.right[label] as f64 / count as f64}),
                )
            })
            .collect();
        let balanced = self
            .classes
            .iter()
            .zip(&self.right)
            .filter(|&(&n, _)| n > 0)
            .map(|(&n, &c)| c as f64 / n as f64)
            .sum::<f64>()
            / classes.len() as f64;
        json!({"loss":self.loss/self.targets as f64,"example_loss":self.example_loss/self.examples as f64,
            "sequence_accuracy":self.exact as f64/self.examples as f64,"token_accuracy":self.correct as f64/self.targets as f64,
            "balanced_accuracy":balanced,"transition_accuracy":if self.transitions>0{Some(self.transition_correct as f64/self.transitions as f64)}else{None},
            "correct":self.correct,"exact":self.exact,"targets":self.targets,"target_count":self.targets,
            "examples":self.examples,"example_count":self.examples,"transition_count":self.transitions,"classes":classes})
    }
}
fn numeric_error(e: &Example, predicted: &[usize]) -> (f64, usize) {
    let mut error = 0.;
    let mut coverage = 0;
    for i in 0..32 {
        let truth = if e.task == "linear-transport" {
            (e.prompt[1 + i] - 52) as f64 / 16.
        } else {
            (e.prompt[1 + i] - 52) as f64 / 4.
                + (e.prompt[33 + i] - 52) as f64 / 16. / (1. + f32::EPSILON as f64).sqrt()
        };
        if let Some(&token) = predicted.get(i).filter(|&&t| (36..69).contains(&t)) {
            error += ((token as f64 - 52.) / 4. - truth).powi(2);
            coverage += 1;
        } else {
            error += 64.;
        }
    }
    (error, coverage)
}
fn field_prediction(logits: &[f64], base: i64) -> usize {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut mass = [0f64; 97];
    for (token, &value) in logits.iter().enumerate() {
        mass[(token as i64 - logits.len() as i64 / 2 + base).rem_euclid(97) as usize] +=
            (value - max).exp();
    }
    argmax(&mass)
}
fn rollout(e: &Example, mut predict: impl FnMut(&[i64]) -> usize) -> Vec<usize> {
    let mut context = e.prompt.clone();
    let mut written = vec![];
    while written.len() < e.limit {
        let token = predict(&context);
        written.push(token);
        if token == 17 {
            break;
        }
        context.push(token as i64);
    }
    written
}
pub fn evaluate(model: &impl Predictor, examples: &[Example]) -> Result<Value> {
    if examples.is_empty() {
        return Err("cannot evaluate empty split".into());
    }
    let vocab = model.vocab();
    let generative = !examples[0].prompt.is_empty();
    let numeric = matches!(
        examples[0].task.as_str(),
        "linear-transport" | "branch-amplitudes"
    );
    let field = examples[0].field.is_some();
    let mut teacher = Counts::new(vocab);
    let mut generated = Counts::new(vocab);
    let (mut generated_tokens, mut ended, mut final_correct) = (0, 0, 0);
    let (mut mse, mut covered, mut native_correct) = (0., 0, 0);
    for e in examples {
        let mut exact = true;
        let mut row_loss = 0.;
        let mut targets = 0;
        let mut written = vec![];
        for (pos, &target) in e.targets.iter().enumerate() {
            if target < 0 {
                continue;
            }
            let label = target as usize;
            let logits = model.logits(&e.tokens[..=pos]);
            let prediction = argmax(&logits);
            let hit = prediction == label;
            exact &= hit;
            row_loss += cross_entropy(&logits, label);
            targets += 1;
            written.push(prediction);
            teacher.observe(label, hit, e.changes[pos]);
        }
        teacher.loss += row_loss;
        teacher.example_loss += row_loss / targets as f64;
        teacher.examples += 1;
        if let Some((base, future)) = e.field {
            native_correct += usize::from(exact);
            exact = field_prediction(&model.logits(&e.tokens), base) as i64 == future;
        }
        teacher.exact += usize::from(exact);
        if numeric && !generative {
            let (error, valid) = numeric_error(e, &written);
            mse += error;
            covered += valid;
        }
        if !generative {
            continue;
        }
        let tokens = rollout(e, |prefix| argmax(&model.logits(prefix)));
        let exact = tokens.len() == e.answer.len()
            && tokens.iter().zip(&e.answer).all(|(&a, &b)| a as i64 == b);
        let eos = tokens.last() == Some(&17);
        ended += usize::from(eos);
        generated_tokens += tokens.len();
        let final_hit = if matches!(e.task.as_str(), "parity" | "boolean_and") {
            eos && tokens.len() >= 2
                && tokens[tokens.len() - 2] as i64 == e.answer[e.answer.len() - 2]
        } else {
            exact
        };
        final_correct += usize::from(final_hit);
        generated.exact += usize::from(exact);
        generated.examples += 1;
        let mut previous = None;
        for (pos, &label) in e.answer.iter().enumerate() {
            let transition = previous != Some(label);
            previous = Some(label);
            let hit = tokens.get(pos).copied().map(|x| x as i64) == Some(label);
            generated.observe(label as usize, hit, transition);
        }
        if numeric {
            let (error, valid) = numeric_error(e, &tokens);
            mse += error;
            covered += valid;
        }
    }
    if !teacher.loss.is_finite() {
        return Err("nonfinite evaluation loss".into());
    }
    let mut result = if generative {
        generated.loss = teacher.loss;
        generated.example_loss = teacher.example_loss;
        let mut result = generated.report();
        result["teacher_forced"] = teacher.report();
        result["final_answer_accuracy"] = (final_correct as f64 / examples.len() as f64).into();
        result["eos_rate"] = (ended as f64 / examples.len() as f64).into();
        result["generated_tokens"] = generated_tokens.into();
        result["extra_tokens"] = 0.into();
        result
    } else {
        teacher.report()
    };
    if field {
        result["decoded_accuracy"] = result["sequence_accuracy"].clone();
        result["token_accuracy"] = result["sequence_accuracy"].clone();
        result["native_token_accuracy"] = (native_correct as f64 / examples.len() as f64).into();
        result["native_correct"] = native_correct.into();
        result["field_correct"] = teacher.exact.into();
        result["correct"] = teacher.exact.into();
    }
    if numeric {
        result["mse"] = (mse / (32 * examples.len()) as f64).into();
        result["numeric_output_coverage"] = (covered as f64 / (32 * examples.len()) as f64).into();
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn live_interface_preserves_every_original_metric() -> Result<()> {
        let a = crate::features::Adapter {
            vocab: 20,
            context: 3,
        };
        let train = [Example::generation("test", vec![1, 18], vec![3, 17])];
        let tests = [Example::generation("test", vec![1, 18], vec![4, 17])];
        let (model, _) = crate::forward::fit(a, &train, 0);
        let model = model.unwrap();
        assert_eq!(
            evaluate(&model, &tests)?,
            crate::reference_metrics::evaluate(&model, &tests)?
        );
        Ok(())
    }
    #[test]
    fn generation_reads_own_errors_and_stops_at_eos_or_limit() {
        let e = Example::generation("test", vec![1, 18], vec![4, 17]);
        let mut calls = 0;
        assert_eq!(
            rollout(&e, |prefix| {
                calls += 1;
                if calls == 1 {
                    assert_eq!(prefix, &[1, 18]);
                    3
                } else {
                    assert_eq!(prefix, &[1, 18, 3]);
                    10
                }
            }),
            vec![3, 10]
        );
        assert_eq!(rollout(&e, |_| 17), vec![17]);
    }
    #[test]
    fn field_pooling_and_numeric_penalties_match_the_suite() {
        let mut logits = vec![-1000.; 256];
        logits[0] = 0.5f64.ln();
        logits[97] = 0.4f64.ln();
        logits[10] = 0.8f64.ln();
        assert_eq!(
            field_prediction(&logits, 8),
            (-128i64 + 8).rem_euclid(97) as usize
        );
        let mut prompt = vec![1];
        prompt.extend([56; 32]);
        prompt.push(18);
        let mut answer = vec![53; 32];
        answer.push(17);
        let e = Example::generation("linear-transport", prompt, answer);
        assert_eq!(numeric_error(&e, &[53; 32]), (0., 32));
        assert_eq!(numeric_error(&e, &[17]), (2048., 0));
    }
}

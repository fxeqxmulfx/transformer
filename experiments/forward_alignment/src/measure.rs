//! Complete terminal measurements. Gradient aggregation uses actual batch
//! counts, including 96/97-row tail batches; held-out derivatives are audits.
use crate::{
    config::Kernels,
    domain::{
        Result,
        data::{Data, Plan},
    },
    infrastructure::engine::Engine,
    inputs::require,
};
use serde_json::{Value, json};

#[derive(Default)]
pub struct Cost {
    pub forward_batches: usize,
    pub backward_batches: usize,
    pub local_output: usize,
    pub hidden_jacobians: usize,
    pub vocabulary_jacobians: usize,
    pub extra_embedding_backwards: usize,
    pub optimizer_steps: usize,
    pub cpu_forward_calls: usize,
    pub cpu_forward_rows: usize,
    pub frozen_local_evaluations: usize,
}
impl Cost {
    pub fn value(&self) -> Value {
        json!({"source_forward_batches":self.forward_batches,
    "terminal_backward_batches":self.backward_batches,"local_output_jacobians":self.local_output,
    "hidden_local_jacobians":self.hidden_jacobians,"vocabulary_jacobians":self.vocabulary_jacobians,
    "extra_embedding_backwards":self.extra_embedding_backwards,"optimizer_steps":self.optimizer_steps,
        "cpu_backbone_forward_calls":self.cpu_forward_calls,"cpu_backbone_forward_rows":self.cpu_forward_rows,
    "cpu_frozen_local_objective_evaluations":self.frozen_local_evaluations})
    }
}

#[derive(Clone, Copy)]
pub enum Scope {
    Batch,
    Train,
    Validation,
}
impl Scope {
    pub const ALL: [Self; 3] = [Self::Batch, Self::Train, Self::Validation];
    pub fn name(self) -> &'static str {
        match self {
            Self::Batch => "batch",
            Self::Train => "train",
            Self::Validation => "validation",
        }
    }
    pub fn validation(self) -> bool {
        matches!(self, Self::Validation)
    }
    pub fn ordered(self) -> bool {
        matches!(self, Self::Batch)
    }
    pub fn bounds(self, plan: &Plan, step: usize) -> (usize, usize) {
        match self {
            Self::Batch => (
                plan.boundaries[step],
                plan.boundaries[step + 1] - plan.boundaries[step],
            ),
            Self::Train => (0, 4704),
            Self::Validation => (0, 4705),
        }
    }
    pub fn rows(self, data: &Data, plan: &Plan, step: usize) -> Vec<u32> {
        match self {
            Self::Train => data.train.clone(),
            Self::Validation => data.validation.clone(),
            Self::Batch => {
                let (start, n) = self.bounds(plan, step);
                plan.order[start..start + n]
                    .iter()
                    .flat_map(|&i| {
                        data.train[3 * i as usize..3 * i as usize + 3]
                            .iter()
                            .copied()
                    })
                    .collect()
            }
        }
    }
}
pub struct Measured {
    pub metrics: Value,
    pub gradients: [Vec<f64>; 2],
}
pub fn terminal(
    engine: &Engine,
    kernels: &Kernels,
    scope: Scope,
    plan: &Plan,
    step: usize,
    gradients: bool,
    cost: &mut Cost,
) -> Result<Measured> {
    let (start, count) = scope.bounds(plan, step);
    let mut g = if gradients {
        [vec![0.; engine.arm.count()], vec![0.; engine.arm.count()]]
    } else {
        [vec![], vec![]]
    };
    let mut loss = [0.; 2];
    let mut correct = 0;
    for offset in (0..count).step_by(512) {
        let batch = (count - offset).min(512);
        engine.forward(scope.validation(), start + offset, batch, scope.ordered())?;
        cost.forward_batches += 1;
        let ce = engine.metrics(batch)?;
        correct += ce.correct;
        loss[0] += ce.loss * batch as f64;
        if gradients {
            engine.backward(batch)?;
            cost.backward_batches += 1;
            add_mean(&mut g[0], &engine.gradients.read()?, batch, count);
        }
        let pair = engine.pair_scores(kernels, batch)?;
        loss[1] += pair.loss * batch as f64;
        if gradients {
            engine.backward(batch)?;
            cost.backward_batches += 1;
            add_mean(&mut g[1], &engine.gradients.read()?, batch, count);
        }
    }
    require(
        g.iter().flatten().all(|v| v.is_finite()),
        "nonfinite full gradient",
    )?;
    Ok(Measured {
        metrics: json!({"ce":loss[0]/count as f64,"pair":loss[1]/count as f64,
        "correct":correct,"count":count,"accuracy":correct as f64/count as f64}),
        gradients: g,
    })
}
fn add_mean(total: &mut [f64], batch: &[f32], rows: usize, count: usize) {
    let weight = rows as f64 / count as f64;
    for (a, &b) in total.iter_mut().zip(batch) {
        *a += weight * b as f64;
    }
}
pub fn metrics_all(
    engine: &Engine,
    kernels: &Kernels,
    plan: &Plan,
    step: usize,
    cost: &mut Cost,
) -> Result<Value> {
    let mut out = json!({});
    for scope in Scope::ALL {
        out[scope.name()] = terminal(engine, kernels, scope, plan, step, false, cost)?.metrics;
    }
    Ok(out)
}
pub fn replay(metrics: &Value, expected: &Value) -> bool {
    metrics["correct"] == expected["correct"]
        && metrics["count"] == expected["count"]
        && expected["loss"]
            .as_f64()
            .is_some_and(|loss| (metrics["ce"].as_f64().unwrap() - loss).abs() <= 1e-12)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uneven_batch_means_match_a_single_mean() {
        let mut g = vec![0., 0.];
        add_mean(&mut g, &[2., -3.], 512, 608);
        add_mean(&mut g, &[-5., 7.], 96, 608);
        assert!((g[0] - (512. * 2. - 96. * 5.) / 608.).abs() < 1e-15);
        assert!((g[1] - (-512. * 3. + 96. * 7.) / 608.).abs() < 1e-15);
    }
}

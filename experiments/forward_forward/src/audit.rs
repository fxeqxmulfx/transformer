//! Fixed source-checkpoint audit; no training benefit is assumed or selected.
use crate::{
    algebra,
    application::Backend,
    domain::{
        Result,
        model::{E, W},
    },
    infrastructure::{engine::Engine, storage},
    inputs::{Case, Saved, require},
    kernel, loss,
};
use serde_json::{Value, json};

#[derive(Default)]
struct Checks {
    rows: usize,
    gradients: usize,
    violations: usize,
    max_loss_error: f64,
    max_gradient_error: f64,
}
impl Checks {
    fn batch(&mut self, logits: &[f32], labels: &[usize], values: &[f32], gradients: &[f32]) {
        for (row, &target) in labels.iter().enumerate() {
            let z: Vec<_> = logits[row * 97..(row + 1) * 97]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let (value, expected) = loss::pairwise(&z, target, 1.);
            let error = (values[row] as f64 - value).abs();
            self.max_loss_error = self.max_loss_error.max(error);
            self.violations += usize::from(!error.is_finite() || error > 2e-5 + 2e-5 * value.abs());
            self.rows += 1;
            for (c, &expected) in expected.iter().enumerate() {
                let expected = expected / labels.len() as f64;
                let error = (gradients[row * 97 + c] as f64 - expected).abs();
                self.max_gradient_error = self.max_gradient_error.max(error);
                self.violations +=
                    usize::from(!error.is_finite() || error > 2e-7 + 2e-5 * expected.abs());
                self.gradients += 1;
            }
        }
    }
    fn value(&self) -> Value {
        json!({"rows":self.rows,"gradient_coordinates":self.gradients,"violations":self.violations,
            "max_loss_absolute_error":self.max_loss_error,"max_gradient_absolute_error":self.max_gradient_error})
    }
}

fn mean_loss(logits: &[f64], labels: &[usize]) -> f64 {
    logits
        .as_chunks::<97>()
        .0
        .iter()
        .zip(labels)
        .map(|(z, &y)| loss::pairwise(z, y, 1.).0)
        .sum::<f64>()
        / labels.len() as f64
}

/// Exercise exact dimensions and numerically extreme logits before source cases.
pub fn kernel_edges(
    cuda: &kernite_cuda::Cuda,
    kernels: &kernel::kernels::Kernels,
) -> Result<Value> {
    let mut checks = Checks::default();
    for batch in [1, 32, 511, 512] {
        let logits: Vec<f32> = (0..batch * 97)
            .map(|i| match (i / 97) % 4 {
                0 => 0.,
                1 => {
                    if i % 97 == 0 {
                        1000.
                    } else {
                        -1000.
                    }
                }
                2 => 10000. + (i % 97) as f32,
                _ => (i % 97) as f32 - 48.,
            })
            .collect();
        let labels: Vec<u32> = (0..batch).map(|i| (i % 97) as u32).collect();
        let z = cuda.upload(&logits)?;
        let y = cuda.upload(&labels)?;
        let values = cuda.alloc::<f32>(batch)?;
        let gradient = cuda.alloc::<f32>(batch * 97)?;
        // SAFETY: every pointer belongs to this context; shapes and labels are valid.
        unsafe {
            kernels.pairwise(
                kernite_cuda::Grid::new(batch.div_ceil(4) as u32, 128),
                gradient.ptr(),
                values.ptr(),
                z.ptr(),
                y.ptr(),
                batch as u32,
            )?;
        }
        checks.batch(
            &logits,
            &labels.iter().map(|&y| y as usize).collect::<Vec<_>>(),
            &values.read()?,
            &gradient.read()?,
        );
    }
    Ok(checks.value())
}

pub fn run(case: &Case<'_>, saved: &Saved, kernel: &kernel::kernels::Kernels) -> Result<Value> {
    let mut engine = Engine::new(case.arm, &saved.state.parameters, case.data, case.plan)?;
    engine.restore(&saved.state)?;
    let mut checks = Checks::default();
    let mut splits = json!({});
    let mut replay = true;
    for (name, validation, rows) in [
        ("train", false, &case.data.train),
        ("validation", true, &case.data.validation),
    ] {
        let count = rows.len() / 3;
        let (mut ce, mut pairwise, mut correct) = (0., 0., 0);
        for offset in (0..count).step_by(512) {
            let batch = (count - offset).min(512);
            engine.forward(validation, offset, batch, false)?;
            let metrics = engine.metrics(batch)?;
            ce += metrics.loss * batch as f64;
            correct += metrics.correct;
            let logits = engine.logits.read()?;
            let labels: Vec<_> = rows[offset * 3..(offset + batch) * 3]
                .as_chunks::<3>()
                .0
                .iter()
                .map(|r| r[2] as usize)
                .collect();
            let (values, gradient) = engine.local_loss(kernel, batch)?;
            checks.batch(&logits, &labels, &values, &gradient);
            pairwise += values.iter().map(|&x| x as f64).sum::<f64>();
        }
        ce /= count as f64;
        pairwise /= count as f64;
        replay &= saved.last_report[name]["correct"] == correct
            && (saved.last_report[name]["loss"].as_f64().unwrap() - ce).abs() < 1e-12;
        splits[name] = json!({"count":count,"correct":correct,"accuracy":correct as f64/count as f64,
            "source_ce":ce,"pairwise_mean_loss":pairwise});
    }
    let offset = case.plan.boundaries[case.step];
    let batch = case.plan.boundaries[case.step + 1] - offset;
    let labels: Vec<_> = case.plan.order[offset..offset + batch]
        .iter()
        .map(|&i| case.data.train[i as usize * 3 + 2] as usize)
        .collect();
    engine.forward(false, offset, batch, true)?;
    let logits: Vec<_> = engine.logits.read()?[..batch * 97]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let features: Vec<_> = engine
        .local_features(batch)?
        .iter()
        .map(|&v| v as f64)
        .collect();
    engine.local_loss(kernel, batch)?;
    let gradient = engine.local_readout_gradient(batch)?;
    // Independent FP64 outer products at the actually executed logits/features.
    let q: Vec<_> = logits
        .as_chunks::<97>()
        .0
        .iter()
        .zip(&labels)
        .flat_map(|(z, &y)| {
            loss::pairwise(z, y, 1.)
                .1
                .into_iter()
                .map(|v| v / batch as f64)
        })
        .collect();
    let qt: Vec<_> = (0..97)
        .flat_map(|c| {
            let q = &q;
            (0..batch).map(move |r| q[r * 97 + c])
        })
        .collect();
    let ht: Vec<_> = (0..W)
        .flat_map(|k| {
            let features = &features;
            (0..batch).map(move |r| features[r * W + k])
        })
        .collect();
    let expected = algebra::linear(&qt, 97, batch, &ht, W);
    let max_error = gradient
        .iter()
        .zip(&expected)
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0., f64::max);
    let max_reference = expected.iter().map(|v| v.abs()).fold(0., f64::max);
    require(
        gradient.iter().all(|v| v.is_finite()),
        "nonfinite local gradient",
    )?;
    let old_e: Vec<_> = saved.state.parameters[..E]
        .iter()
        .map(|&x| x as f64)
        .collect();
    let frozen_before = mean_loss(&algebra::linear(&features, batch, W, &old_e, 97), &labels);
    let mut changed = engine.state()?;
    for (p, &g) in changed.parameters[..E].iter_mut().zip(&gradient) {
        *p -= 1e-3f32 * g;
    }
    let changed_coordinates = changed.parameters[..E]
        .iter()
        .zip(&saved.state.parameters[..E])
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let new_e: Vec<_> = changed.parameters[..E].iter().map(|&x| x as f64).collect();
    let frozen_after = mean_loss(&algebra::linear(&features, batch, W, &new_e, 97), &labels);
    engine.restore(&changed)?;
    engine.forward(false, offset, batch, true)?;
    let tied_after: Vec<_> = engine.logits.read()?[..batch * 97]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let tied_before = mean_loss(&logits, &labels);
    let tied_after = mean_loss(&tied_after, &labels);
    engine.restore(&saved.state)?;
    require(
        storage::state_bytes(case.step, &engine.state()?)
            == storage::state_bytes(case.step, &saved.state),
        "source state not restored after local probe",
    )?;
    Ok(
        json!({"passed":replay && checks.violations==0 && max_error <= 2e-6+2e-4*max_reference,
        "saved_metrics_replayed":replay,"kernel":checks.value(),"splits":splits,
        "partial_readout":{"batch":batch,"max_gradient_error":max_error,"max_gradient_reference":max_reference,
            "gradient_squared_norm":expected.iter().map(|v|v*v).sum::<f64>(),"changed_coordinates":changed_coordinates,
            "frozen_features_before":frozen_before,"frozen_features_after":frozen_after,"frozen_features_change":frozen_after-frozen_before,
            "actual_tied_before":tied_before,"actual_tied_after":tied_after,"actual_tied_change":tied_after-tied_before,
            "state_restored":true,"step_size":0.001}}),
    )
}

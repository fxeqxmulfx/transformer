use super::*;
use tch::{Device, Kind, Tensor, nn};

fn difference(a: &Tensor, b: &Tensor) -> f64 {
    (a - b).abs().max().double_value(&[])
}

#[test]
fn closed_gates_restart_from_prelude_and_open_gates_preserve_causality() -> Result<()> {
    tch::set_num_threads(1);
    let mut engine = engine::Engine::new(16, 4, 9, 1)?;
    let dense = (Tensor::arange(5 * 16, (Kind::Float, Device::Cpu)) * 0.17)
        .sin()
        .view([1, 5, 16]);
    let first = engine.model.inner(&dense, 1);
    assert_eq!(difference(&first, &engine.model.inner(&dense, 3)), 0.);
    tch::no_grad(|| {
        for (name, mut p) in engine.variables.variables() {
            if name.ends_with(".admit") || name.ends_with(".forget") {
                let _ = p.fill_(if name.ends_with(".admit") { 0.4 } else { -0.1 });
            }
        }
    });
    let next = engine.model.inner(&dense, 2);
    let last = engine.model.inner(&dense, 3);
    assert!(difference(&first, &next) > 1e-5 && difference(&next, &last) > 1e-7);
    let altered = Tensor::cat(&[dense.narrow(1, 0, 3), -dense.narrow(1, 3, 2)], 1);
    assert_eq!(
        difference(
            &last.narrow(1, 0, 3),
            &engine.model.inner(&altered, 3).narrow(1, 0, 3)
        ),
        0.
    );
    let tokens = [2, 4, 5, 8];
    engine.forward(&tokens, &[5, 4, 3, 0], 1, 4)?;
    let a = engine.logits.read()?;
    engine.forward(&tokens, &[u32::MAX, 0, 1, 2], 1, 4)?;
    assert_eq!(a, engine.logits.read()?);
    Ok(())
}

#[test]
fn shared_core_backpropagates_through_three_passes() -> Result<()> {
    tch::set_num_threads(1);
    let engine = engine::Engine::new(16, 4, 9, 2)?;
    let params = engine.variables.variables();
    tch::no_grad(|| {
        for (name, value) in &params {
            if name.ends_with(".admit") {
                let _ = value.shallow_clone().fill_(0.3);
            }
            // Keep the selected experts constant on both sides of this finite
            // difference. The dispatch test checks the expert and router-weight adjoints.
            if name.ends_with(".router") {
                let _ = value.shallow_clone().zero_();
            }
        }
    });
    let ids = Tensor::from_slice(&[1i64, 2, 4, 5]).view([1, 4]);
    let objective = || engine.model.logits(&ids).select(-1, 3).sum(Kind::Float);
    let weight = &params[&format!("layer{}.attn.out", text::GROUP + 1)];
    objective().backward();
    let gradient = weight.grad().view([-1]);
    let coordinate = gradient.abs().argmax(None, false).int64_value(&[]);
    let analytic = gradient.double_value(&[coordinate]);
    let mut cell = weight.view([-1]).get(coordinate);
    let original = cell.double_value(&[]);
    let eps = 5e-4;
    let numeric = tch::no_grad(|| {
        let _ = cell.fill_(original + eps);
        let plus = objective().double_value(&[]);
        let _ = cell.fill_(original - eps);
        let minus = objective().double_value(&[]);
        let _ = cell.fill_(original);
        (plus - minus) / (2. * eps)
    });
    assert!(
        analytic.abs() > 1e-6 && (numeric - analytic).abs() < 0.004 + 0.05 * analytic.abs(),
        "shared weight: analytic {analytic}, numerical {numeric}"
    );
    Ok(())
}

#[test]
fn kernite_adamw_agrees_with_scalar_updates_including_tail() -> Result<()> {
    let vs = nn::VarStore::new(Device::Cpu);
    let value = vs.root().ones("weight", &[11]);
    let mut adam = optimizer::Adam::new(&vs)?;
    let mut reference = [1f64; 11];
    let mut first = [0f64; 11];
    let mut second = [0f64; 11];
    for step in 0..4 {
        let gradients: Vec<f32> = (0..11).map(|i| ((i + step) % 7) as f32 - 2.).collect();
        (&value * Tensor::from_slice(&gradients))
            .sum(Kind::Float)
            .backward();
        let rate = if step == 0 { 0. } else { 0.002 };
        adam.step(rate as f32, step)?;
        let norm = gradients
            .iter()
            .map(|&g| f64::from(g).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(1.);
        for i in 0..11 {
            let g = f64::from(gradients[i]) / norm;
            first[i] = 0.9 * first[i] + 0.1 * g;
            second[i] = 0.95 * second[i] + 0.05 * g * g;
            let m = first[i] / (1. - 0.9f64.powi(step as i32 + 1));
            let v = second[i] / (1. - 0.95f64.powi(step as i32 + 1));
            reference[i] = reference[i] * (1. - 0.1 * rate) - rate * m / (v.sqrt() + 1e-8);
            assert!((value.double_value(&[i as i64]) - reference[i]).abs() < 4e-7);
        }
    }
    Ok(())
}

#[test]
fn suite_uses_pinned_gpt_metrics_and_schedule() {
    assert_eq!(
        hash(SCORER.as_bytes()),
        "8a64672a3274c7cf56fad2fe6630d264331083a217c05d1de668c1204f59985b"
    );
    assert_eq!(
        hash(include_bytes!(
            "../../../.lake/reproduction/gpt/research/main_model/wsd_suite_native/src/schedule.rs"
        )),
        "5853b9f7255428ef8fa7d11b7e98c329dc0129e0990c47430a77eeaef5c22bd4"
    );
}

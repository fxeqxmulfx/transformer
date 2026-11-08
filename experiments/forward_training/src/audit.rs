//! Predeclared numerical audit: finite differences hold the specified
//! boundaries fixed. Audit gradients and trained checkpoints never train a run.
use crate::{
    application::Backend,
    config::{Aux, Kernels, Method, Prototypes},
    domain::{
        Result,
        data::{Data, Plan, Random},
        model::{self, Arm, E, W},
    },
    infrastructure::{engine::Engine, storage},
    inputs::{self, Ledger, require},
    reference::{self, Cache, Heads, Objective},
};
use kernite_cuda::{Cuda, Grid};
use serde_json::{Value, json};
use std::{ops::Range, path::Path};

fn tolerance(a: f64, b: f64) -> f64 {
    2e-5 + 2e-3 * a.abs().max(b.abs())
}
fn direction(arm: Arm, range: Range<usize>, seed: u64) -> Vec<f64> {
    let mut rng = Random(seed);
    let mut d = vec![0.; arm.count()];
    let mut draw = || if rng.next() & 1 == 0 { 1. } else { -1. };
    if arm == Arm::Swiglu || range.end <= E {
        for k in range {
            d[k] = draw();
        }
    } else {
        for [ar, bi, br, ai, kind] in model::pairs() {
            if kind == 0 && range.contains(&(ar as usize)) {
                let (a, b) = (draw(), draw());
                d[ar as usize] = a;
                d[ai as usize] = a;
                d[bi as usize] = b;
                d[br as usize] = -b;
            }
        }
        if range.start == 0 {
            for value in &mut d[..E] {
                *value = draw();
            }
        }
    }
    let norm = d.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(norm > 0.);
    for v in &mut d {
        *v /= norm;
    }
    d
}

fn gradients(
    case: &inputs::Case<'_>,
    saved: &inputs::Saved,
    cuda: &Cuda,
    kernels: &Kernels,
) -> Result<Value> {
    let arm = case.arm;
    let p: Vec<_> = saved.state.parameters.iter().map(|&x| x as f64).collect();
    let rows = &case.data.train[..24];
    let old = Cache::new(arm, &p, rows);
    let prototypes = Prototypes::new(arm, case.seed);
    let heads = Heads::from(&prototypes);
    let aux = Aux::new(cuda, arm, &prototypes, Method::Auxiliary)?;
    let mut engine = Engine::new(arm, &saved.state.parameters, case.data, case.plan)?;
    engine.restore(&saved.state)?;
    engine.forward(false, 0, 8, false)?;
    engine.local_output(kernels, 8)?;
    let readout = engine.gradients.read()?;
    engine.hidden_objective(&aux, kernels, 8, false)?;
    engine.vocabulary_objective(&aux, kernels)?;
    let local = engine.gradients.read()?;
    let off = E + 2 * arm.hidden() * W;
    let mut records = vec![];
    let mut energy = vec![];
    let mut check = |kind: Objective, g: &[f32], range: Range<usize>| {
        for repetition in 0..2 {
            let d = direction(
                arm,
                range.clone(),
                case.seed ^ 0x6666617564697400 ^ repetition,
            );
            let plus: Vec<_> = p.iter().zip(&d).map(|(p, d)| p + 1e-4 * d).collect();
            let minus: Vec<_> = p.iter().zip(&d).map(|(p, d)| p - 1e-4 * d).collect();
            let finite = (reference::objective(kind, arm, &plus, rows, &old, &heads)
                - reference::objective(kind, arm, &minus, rows, &old, &heads))
                / (2e-4);
            let analytic = g.iter().zip(&d).map(|(&g, d)| g as f64 * d).sum::<f64>();
            let error = (finite - analytic).abs();
            records.push(json!({"objective":format!("{kind:?}"),"direction":repetition,
                "finite":finite,"analytic":analytic,"absolute_error":error,
                "tolerance":tolerance(finite,analytic),"passed":error.is_finite() && error<=tolerance(finite,analytic)}));
        }
    };
    let mut vocabulary = local.clone();
    for k in 0..E {
        vocabulary[k] -= readout[k];
    }
    check(Objective::Readout, &readout, 0..E);
    check(Objective::Vocabulary, &vocabulary, 0..E);
    check(Objective::SharedE, &local, 0..E);
    check(Objective::A, &local, E..off);
    check(Objective::B, &local, off..arm.count());
    for method in [Method::Local, Method::Pair, Method::Auxiliary] {
        engine.restore(&saved.state)?;
        engine.forward(false, 0, 8, false)?;
        engine.pilot_gradient(method, Some(&aux), kernels, 8)?;
        let gradient = engine.gradients.read()?;
        if method != Method::Local {
            check(
                if method == Method::Pair {
                    Objective::Pair
                } else {
                    Objective::Auxiliary
                },
                &gradient,
                0..arm.count(),
            );
        }
        let norms: Vec<_> = [0..E, E..off, off..arm.count()]
            .iter()
            .map(|r| {
                gradient[r.clone()]
                    .iter()
                    .map(|&x| (x as f64).powi(2))
                    .sum::<f64>()
            })
            .collect();
        engine.update(case.step + 1)?;
        let next = engine.state()?;
        let valid = crate::config::valid_state(arm, &next);
        energy.push(json!({"method":method.name(),"squared_norms_E_A_B":norms,
            "valid_updated_state":valid,"passed":valid && norms.iter().all(|&n|n.is_finite() && n>0.)}));
    }
    let offset = case.plan.boundaries[case.step];
    let batch = case.plan.boundaries[case.step + 1] - offset;
    engine.restore(&saved.state)?;
    engine.pilot_step(Method::Ce, None, kernels, offset, batch, case.step + 1)?;
    let candidate = storage::state_bytes(case.step + 1, &engine.state()?);
    engine.restore(&saved.state)?;
    engine.train_step(offset, batch, case.step + 1)?;
    let same = candidate == storage::state_bytes(case.step + 1, &engine.state()?);
    let passed = same
        && records.iter().all(|v| v["passed"] == true)
        && energy.iter().all(|v| v["passed"] == true);
    Ok(
        json!({"passed":passed,"directions":records,"gradient_and_update_checks":energy,
        "source_update_bitwise_match":same,"prototype0_sha256":storage::hash(&storage::floats(&prototypes.vocabulary)),
        "prototype1_sha256":storage::hash(&storage::floats(&prototypes.hidden))}),
    )
}

fn normalization(cuda: &Cuda, kernels: &Kernels) -> Result<Value> {
    let mut records = vec![];
    for width in [128, 341, 682] {
        let rows = 5;
        let mut x = vec![0f32; rows * width];
        let mut dy = vec![0f32; x.len()];
        for r in 0..rows {
            for k in 0..width {
                x[r * width + k] = match r {
                    0 => 0.,
                    1 => 1e-10 * (k as f32 - 3.),
                    2 => (k as f32 % 7. - 3.) * 0.2,
                    3 => (k as f32 % 5. - 2.) * 1e4,
                    _ => {
                        if k == width - 1 {
                            1.
                        } else {
                            0.
                        }
                    }
                };
                dy[r * width + k] = (k as f32 % 11. - 5.) * 0.01;
            }
        }
        let gx = cuda.upload(&x)?;
        let gg = cuda.upload(&dy)?;
        let gy = cuda.alloc::<f32>(x.len())?;
        let gd = cuda.alloc::<f32>(x.len())?;
        let grid = Grid::new(rows.div_ceil(4) as u32, 128);
        // SAFETY: matched buffers, positive fixed dimensions and partial-warp guards.
        unsafe {
            match width {
                128 => {
                    kernels
                        .norm
                        .rms::<128>(grid, gy.ptr(), gx.ptr(), rows as u32)?;
                    kernels.norm.derivative::<128>(
                        grid,
                        gd.ptr(),
                        gx.ptr(),
                        gg.ptr(),
                        rows as u32,
                    )?;
                }
                341 => {
                    kernels
                        .norm
                        .rms::<341>(grid, gy.ptr(), gx.ptr(), rows as u32)?;
                    kernels.norm.derivative::<341>(
                        grid,
                        gd.ptr(),
                        gx.ptr(),
                        gg.ptr(),
                        rows as u32,
                    )?;
                }
                _ => {
                    kernels
                        .norm
                        .rms::<682>(grid, gy.ptr(), gx.ptr(), rows as u32)?;
                    kernels.norm.derivative::<682>(
                        grid,
                        gd.ptr(),
                        gx.ptr(),
                        gg.ptr(),
                        rows as u32,
                    )?;
                }
            }
        }
        let y = gy.read()?;
        let d = gd.read()?;
        let mut errors = [0f64; 2];
        let mut violations = 0;
        for r in 0..rows {
            let x: Vec<_> = x[r * width..(r + 1) * width]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let g: Vec<_> = dy[r * width..(r + 1) * width]
                .iter()
                .map(|&x| x as f64)
                .collect();
            let s = (x.iter().map(|v| v * v).sum::<f64>() / width as f64 + reference::EPS).sqrt();
            let dot = x.iter().zip(&g).map(|(x, g)| x * g).sum::<f64>() / width as f64;
            for k in 0..width {
                for (j, actual, expected) in [
                    (0, y[r * width + k] as f64, x[k] / s),
                    (
                        1,
                        d[r * width + k] as f64,
                        g[k] / s - x[k] * dot / s.powi(3),
                    ),
                ] {
                    let e = (actual - expected).abs();
                    errors[j] = errors[j].max(e);
                    violations += usize::from(!e.is_finite() || e > tolerance(actual, expected));
                }
            }
        }
        records.push(json!({"width":width,"rows":rows,"max_forward_error":errors[0],"max_derivative_error":errors[1],"violations":violations}));
    }
    Ok(json!({"passed":records.iter().all(|v|v["violations"]==0),"cases":records}))
}

pub fn run(
    root: &Path,
    output: &Path,
    ledger: &mut Ledger,
    protocol: &str,
    cuda: &Cuda,
    kernels: &Kernels,
) -> Result<Value> {
    let norm = normalization(cuda, kernels)?;
    let mut records = vec![];
    for seed in [7, 11] {
        let data = Data::new(seed);
        let plan = Plan::new(seed, 10001);
        for arm in Arm::ALL {
            for step in [0, 5000, 10000] {
                let case = inputs::Case {
                    arm,
                    seed,
                    step,
                    data: &data,
                    plan: &plan,
                    protocol,
                    forecast_output: output,
                };
                let saved = inputs::checkpoint(ledger, root, &case)?;
                if step == 10000 {
                    continue;
                }
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    gradients(&case, &saved, cuda, kernels)
                }));
                let mut record = json!({"arm":arm.name(),"seed":seed,"step":step});
                match outcome {
                    Ok(Ok(value)) => {
                        record["passed"] = value["passed"].clone();
                        record["audit"] = value;
                    }
                    Ok(Err(error)) => {
                        record["passed"] = json!(false);
                        record["error"] = json!(error.to_string());
                    }
                    Err(_) => {
                        record["passed"] = json!(false);
                        record["error"] = json!("panic in numerical preflight");
                    }
                }
                eprintln!(
                    "preflight {} seed={} step={} {}",
                    arm.name(),
                    seed,
                    step,
                    record["passed"]
                );
                records.push(record);
            }
        }
    }
    let passed = norm["passed"] == true && records.iter().all(|v| v["passed"] == true);
    let report = json!({"passed":passed,"normalization":norm,"cases":records});
    storage::json_write(output, &report)?;
    require(
        passed,
        "preflight failed; details retained, training not started",
    )?;
    Ok(report)
}

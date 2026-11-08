//! Independent coordinate update oracle and exact source E/B comparisons.
use crate::{
    application::{Backend, State},
    config::{self, Aux, Kernels, Method, Prototypes},
    coordinates as coord,
    domain::{
        Result,
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::engine::Engine,
    normalized::{self, Normalizer},
    normalized_kernel,
    reference::{self, Cache, Heads, Objective},
};
use kernite_cuda::Cuda;
use serde_json::{Value, json};
pub struct Case<'a> {
    pub arm: Arm,
    pub seed: u64,
    pub step: usize,
    pub state: &'a State,
    pub data: &'a Data,
    pub plan: &'a Plan,
}
fn bits(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits())
}
fn compare(
    arm: Arm,
    p: &[f32],
    actual: &[f32],
    expected: &[f32],
    stats: &[f64],
    reference: &[f64; 3],
) -> Value {
    let range = coord::expanded_groups(arm)[1].1.clone();
    let mut maximum = 0f64;
    let mut violations = 0;
    for i in range.clone() {
        let e = (actual[i] as f64 - expected[i] as f64).abs();
        maximum = maximum.max(e);
        violations += usize::from(!e.is_finite() || e > 2e-8 + 2e-6 * (expected[i] as f64).abs());
    }
    let errors: Vec<_> = reference
        .iter()
        .zip(stats)
        .map(|(&r, &a)| (r - a).abs())
        .collect();
    let norms = errors
        .iter()
        .zip(reference)
        .all(|(&e, &r)| e.is_finite() && e <= 1e-12 * (1. + r.abs()));
    let outside = (0..p.len()).all(|i| range.contains(&i) || actual[i].to_bits() == p[i].to_bits());
    json!({"max_parameter_error":maximum,"coordinate_violations":violations,"norm_errors":errors,
        "norms_passed":norms,"valid_ties":model::valid(arm,actual),"outside_A_unchanged":outside,
        "passed":violations==0&&norms&&model::valid(arm,actual)})
}
pub fn actual(
    case: &Case<'_>,
    cuda: &Cuda,
    kernels: &Kernels,
    normalized: &normalized_kernel::kernels::Kernels,
) -> Result<Value> {
    let Case {
        arm,
        seed,
        step,
        state,
        data,
        plan,
    } = *case;
    let mut engine = Engine::new(arm, &state.parameters, data, plan)?;
    engine.restore(state)?;
    let prototypes = Prototypes::new(arm, seed);
    let aux = Aux::new(cuda, arm, &prototypes, Method::Local)?;
    let normalizer = Normalizer::new(cuda, arm)?;
    let mut derivatives = vec![];
    if step == 0 {
        engine.forward(false, 0, 8, false)?;
        engine.pilot_gradient(Method::Local, Some(&aux), kernels, 8)?;
        let g = engine.gradients.read()?;
        let p: Vec<_> = state.parameters.iter().map(|&v| v as f64).collect();
        let rows = &data.train[..24];
        let old = Cache::new(arm, &p, rows);
        let heads = Heads::from(&prototypes);
        for repetition in 0..2 {
            let direction = coord::direction(1, seed ^ 0x6e6f726d616c0000 ^ repetition);
            let expanded = coord::expand(arm, &direction);
            let plus: Vec<_> = p.iter().zip(&expanded).map(|(p, d)| p + 1e-4 * d).collect();
            let minus: Vec<_> = p.iter().zip(&expanded).map(|(p, d)| p - 1e-4 * d).collect();
            let finite = (reference::objective(Objective::A, arm, &plus, rows, &old, &heads)
                - reference::objective(Objective::A, arm, &minus, rows, &old, &heads))
                / 2e-4;
            let analytic = g
                .iter()
                .zip(&expanded)
                .map(|(&g, d)| g as f64 * d)
                .sum::<f64>();
            let error = (finite - analytic).abs();
            let tolerance = 2e-5 + 2e-3 * finite.abs().max(analytic.abs());
            derivatives.push(json!({"direction":repetition,"finite":finite,"analytic":analytic,"error":error,"tolerance":tolerance,"passed":error.is_finite()&&error<=tolerance}));
        }
    }
    let offset = plan.boundaries[step];
    let batch = plan.boundaries[step + 1] - offset;
    engine.forward(false, offset, batch, true)?;
    engine.pilot_gradient(Method::Local, Some(&aux), kernels, batch)?;
    let gradient = engine.gradients.read()?;
    let (expected, norms) = normalized::reference(arm, &state.parameters, &gradient, step + 1);
    engine.normalized_update(step + 1, &normalizer, normalized)?;
    let candidate = engine.state()?;
    let stats = normalizer.stats.read()?;
    let checked = compare(
        arm,
        &state.parameters,
        &candidate.parameters,
        &expected,
        &stats,
        &norms,
    );
    engine.restore(state)?;
    engine.pilot_step(Method::Local, Some(&aux), kernels, offset, batch, step + 1)?;
    let original = engine.state()?;
    let a = coord::expanded_groups(arm)[1].1.clone();
    let eb = [
        (&candidate.parameters, &original.parameters),
        (&candidate.first, &original.first),
        (&candidate.second, &original.second),
    ]
    .iter()
    .all(|(x, y)| (0..x.len()).all(|i| a.contains(&i) || x[i].to_bits() == y[i].to_bits()));
    let moments = bits(&candidate.first[a.clone()], &state.first[a.clone()])
        && bits(&candidate.second[a.clone()], &state.second[a]);
    let valid = config::valid_state(arm, &candidate);
    let passed = checked["passed"] == true
        && eb
        && moments
        && valid
        && derivatives.iter().all(|r| r["passed"] == true);
    Ok(
        json!({"passed":passed,"update":checked,"E_B_bitwise_source_match":eb,"A_moments_unchanged":moments,"valid_state":valid,"norm_stats":stats,"directions":derivatives}),
    )
}
pub fn edges(cuda: &Cuda, kernels: &normalized_kernel::kernels::Kernels) -> Result<Value> {
    let mut records = vec![];
    let data = Data::new(17);
    let plan = Plan::new(17, 1);
    for arm in Arm::ALL {
        for kind in ["zero-gradient", "zero-A", "signed-nonuniform"] {
            let mut p = model::initialize(arm, 17);
            if kind == "zero-A" {
                for i in coord::expanded_groups(arm)[1].1.clone() {
                    p[i] = 0.;
                }
            }
            let mut g: Vec<_> = (0..arm.count())
                .map(|i| {
                    if kind == "zero-gradient" {
                        0.
                    } else {
                        (i % 13) as f32 * 0.001 - 0.006
                    }
                })
                .collect();
            if kind == "signed-nonuniform" {
                g[coord::expanded_groups(arm)[1].1.end - 1] = 0.17;
            }
            let mut engine = Engine::new(arm, &p, &data, &plan)?;
            engine.gradients.write(&g)?;
            let normalizer = Normalizer::new(cuda, arm)?;
            engine.normalized_a_only(10, &normalizer, kernels)?;
            let actual = engine.parameters.read()?;
            let stats = normalizer.stats.read()?;
            let (expected, norms) = normalized::reference(arm, &p, &g, 10);
            let comparison = compare(arm, &p, &actual, &expected, &stats, &norms);
            let zero_ok = if kind == "signed-nonuniform" {
                stats[3] == 0.
            } else {
                stats[3] == 1. && bits(&p, &actual)
            };
            records.push(json!({"arm":arm.name(),"kind":kind,"passed":comparison["passed"]==true&&comparison["outside_A_unchanged"]==true&&zero_ok,"update":comparison,"zero_case_passed":zero_ok,"stats":stats}));
        }
    }
    Ok(json!({"passed":records.iter().all(|r|r["passed"]==true),"cases":records}))
}

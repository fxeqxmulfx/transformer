//! Read-only source forward/backward probes and the unchanged GPU clipping norm.
use crate::{
    Result,
    application::Backend,
    domain::{
        data::{Data, Plan},
        model::{self, Arm},
    },
    evidence::require,
    infrastructure::engine::Engine,
    optimizer_kernels::kernels,
};
use kernite_cuda::{Cuda, Grid};
pub struct Snapshot {
    pub features: Vec<f64>,
    pub logits: Vec<f64>,
}
pub fn engine(state: &crate::application::State, data: &Data, plan: &Plan) -> Result<Engine> {
    let mut e = Engine::new(Arm::Plain, &state.parameters, data, plan)?;
    e.restore(state)?;
    Ok(e)
}
pub fn snapshot(e: &Engine, data: &Data) -> Result<Snapshot> {
    let n = data.train.len() / 3;
    let mut out = Snapshot {
        features: Vec::with_capacity(n * 128),
        logits: Vec::with_capacity(n * 97),
    };
    for at in (0..n).step_by(512) {
        let batch = (n - at).min(512);
        e.forward(false, at, batch, false)?;
        let (_, features) = e.audit_features(batch)?;
        out.features.extend(features);
        out.logits
            .extend(e.logits.read()?[..batch * 97].iter().map(|x| *x as f64));
    }
    require(
        out.features
            .iter()
            .chain(&out.logits)
            .all(|v| v.is_finite()),
        "nonfinite execution",
    )?;
    Ok(out)
}
pub fn minibatch(
    e: &Engine,
    data: &Data,
    plan: &Plan,
    anchor: usize,
) -> Result<crate::metric_analysis::Executed> {
    let at = plan.boundaries[anchor];
    let batch = plan.boundaries[anchor + 1] - at;
    e.forward(false, at, batch, true)?;
    let (_, features) = e.audit_features(batch)?;
    let (partial, full) = e.audit_gradient(batch)?;
    let cuda = Cuda::new()?;
    let kernels = kernels::Kernels::load(&cuda)?;
    let owners = model::pairs();
    let map = cuda.upload(owners.as_flattened())?;
    let gradient = cuda.alloc::<f32>(2 * owners.len())?;
    let parts = cuda.alloc::<f32>(gradient.len().div_ceil(256))?;
    let norm = cuda.alloc::<f32>(1)?;
    // SAFETY: identical source ownership map and buffer/grid sizes as optimizer.rs.
    // These three kernels compute a norm only; no parameter or moment is updated.
    unsafe {
        kernels.contract(
            Grid::new(owners.len().div_ceil(256) as u32, 256),
            gradient.ptr(),
            e.gradients.ptr(),
            map.ptr(),
            owners.len() as u32,
        )?;
        kernels.norm_partial(
            Grid::new(parts.len() as u32, 32),
            parts.ptr(),
            gradient.ptr(),
            gradient.len() as u32,
        )?;
        kernels.norm_finish(
            Grid::new(1, 32),
            norm.ptr(),
            parts.ptr(),
            parts.len() as u32,
        )?;
    }
    let actual = norm.read()?[0] as f64;
    let raw = e.gradients.read()?;
    let expected = owners
        .iter()
        .flat_map(|&[ar, bi, br, ai, kind]| {
            let a = raw[ar as usize];
            let b = raw[bi as usize];
            if kind == 0 {
                [a + raw[ai as usize], b - raw[br as usize]]
            } else {
                [a, b]
            }
        })
        .map(|v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    require(
        actual.is_finite() && (actual - expected).abs() <= 2e-6 * expected.max(1.),
        "GPU gradient norm",
    )?;
    let rows = plan.order[at..at + batch]
        .iter()
        .flat_map(|i| {
            data.train[*i as usize * 3..(*i as usize + 1) * 3]
                .iter()
                .copied()
        })
        .collect();
    Ok(crate::metric_analysis::Executed {
        features,
        partial,
        full,
        rows,
        norm: actual,
    })
}

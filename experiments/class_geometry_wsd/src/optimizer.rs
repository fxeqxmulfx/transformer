//! Adapter of GPT phase_score_native/src/optimizer.rs at 6bb9d03a.
//! The destination has the same complex ownership convention and no frequency
//! group. Kernels are copied byte-for-byte; only Engine buffer access differs.
use crate::{
    Result,
    application::{Backend, State},
    domain::{
        data::{Data, Plan},
        model::{self, Arm},
    },
    infrastructure::engine::Engine,
    inputs::require,
    optimizer_kernels::kernels,
    schedule::Schedule,
};
use kernite_cuda::{Buffer, Cuda, Grid};
use serde_json::{Value, json};

pub struct Optimizer {
    schedule: Schedule,
    kernels: kernels::Kernels,
    map: Buffer<u32>,
    gradient: Buffer<f32>,
    parts: Buffer<f32>,
    norm: Buffer<f32>,
}
impl Optimizer {
    pub fn new() -> Result<Self> {
        let cuda = Cuda::new()?;
        let owners = model::pairs();
        Ok(Self {
            schedule: Schedule::new(10000)?,
            kernels: kernels::Kernels::load(&cuda)?,
            map: cuda.upload(owners.as_flattened())?,
            gradient: cuda.alloc(2 * owners.len())?,
            parts: cuda.alloc((2 * owners.len()).div_ceil(256))?,
            norm: cuda.alloc(1)?,
        })
    }
    pub fn step(&self, e: &Engine, completed: usize) -> Result<f32> {
        require(e.arm == Arm::Plain, "unsupported optimizer destination")?;
        let count = self.gradient.len() / 2;
        let grid = Grid::new(count.div_ceil(256) as u32, 256);
        let t = (completed + 1) as f64;
        let (first, second) = e.wsd_moments();
        // SAFETY: source pairs cover each independent coordinate exactly once;
        // buffers and scratch sizes match the unchanged kernel launch contract.
        unsafe {
            self.kernels.contract(
                grid,
                self.gradient.ptr(),
                e.gradients.ptr(),
                self.map.ptr(),
                count as u32,
            )?;
            self.kernels.norm_partial(
                Grid::new(self.parts.len() as u32, 32),
                self.parts.ptr(),
                self.gradient.ptr(),
                self.gradient.len() as u32,
            )?;
            self.kernels.norm_finish(
                Grid::new(1, 32),
                self.norm.ptr(),
                self.parts.ptr(),
                self.parts.len() as u32,
            )?;
            self.kernels.adam_coordinates(
                grid,
                e.parameters.ptr(),
                first,
                second,
                self.gradient.ptr(),
                self.map.ptr(),
                self.norm.ptr(),
                count as u32,
                self.schedule.rate(completed),
                0.1,
                (1. - 0.9f64.powf(t)) as f32,
                (1. - 0.95f64.powf(t)).sqrt() as f32,
            )?;
        }
        let norm = self.norm.read()?[0];
        require(norm.is_finite() && norm >= 0., "invalid gradient norm")?;
        Ok(norm)
    }
}

/// Independent contraction into the parameter ownership map, before clipping.
pub fn contracted(g: &[f32]) -> Vec<f32> {
    model::pairs()
        .iter()
        .flat_map(|&[ar, bi, br, ai, kind]| {
            let (a, b) = (g[ar as usize], g[bi as usize]);
            if kind == 0 {
                [a + g[ai as usize], b - g[br as usize]]
            } else {
                [a, b]
            }
        })
        .collect()
}
fn scalar(p: f32, m: f32, v: f32, g: f32, lr: f32, b1: f32, b2: f32) -> [f32; 3] {
    let m = 0.1f32.mul_add(g - m, m);
    let v = 0.05f32.mul_add(g * g, 0.95 * v);
    let p = p * (1. - lr * 0.1) - ((lr / b1) * m) / (v.sqrt() / b2 + 1e-8);
    [p, m, v]
}
/// CPU FP32 reference preserves each operation and repairs the same ties.
pub fn reference(old: &State, g: &[f32], norm: f32, completed: usize) -> State {
    let mut out = State {
        parameters: old.parameters.clone(),
        first: old.first.clone(),
        second: old.second.clone(),
    };
    let grads = contracted(g);
    let lr = Schedule::new(10000).unwrap().rate(completed);
    let b1 = (1. - 0.9f64.powf((completed + 1) as f64)) as f32;
    let b2 = (1. - 0.95f64.powf((completed + 1) as f64)).sqrt() as f32;
    for (i, [ar, bi, br, ai, kind]) in model::pairs().into_iter().enumerate() {
        for (j, at) in [ar, bi].into_iter().enumerate() {
            let k = at as usize;
            let [p, m, v] = scalar(
                old.parameters[k],
                old.first[k],
                old.second[k],
                grads[2 * i + j] / norm.max(1.),
                lr,
                b1,
                b2,
            );
            out.parameters[k] = p;
            out.first[k] = m;
            out.second[k] = v;
        }
        if kind == 0 {
            for (dst, src, sign) in [(ai, ar, 1.), (br, bi, -1.)] {
                out.parameters[dst as usize] = sign * out.parameters[src as usize];
                out.first[dst as usize] = sign * out.first[src as usize];
                out.second[dst as usize] = out.second[src as usize];
            }
        }
    }
    out
}
fn ulps(a: f32, b: f32) -> u32 {
    let ordered = |x: f32| {
        let bits = x.to_bits();
        if bits >> 31 == 0 {
            bits | 0x80000000
        } else {
            !bits
        }
    };
    if a == b {
        0
    } else {
        ordered(a).abs_diff(ordered(b))
    }
}
pub fn check(old: &State, new: &State, raw: &[f32], norm: f32, completed: usize) -> Result<Value> {
    let g = contracted(raw);
    let norm64 = g.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    let norm_error = (norm as f64 - norm64).abs() / norm64.max(1.);
    require(norm_error <= 2e-6, "independent-coordinate norm mismatch")?;
    let expected = reference(old, raw, norm, completed);
    let mut maximum = 0;
    for (actual, want) in [&new.parameters, &new.first, &new.second].into_iter().zip([
        &expected.parameters,
        &expected.first,
        &expected.second,
    ]) {
        for (&a, &b) in actual.iter().zip(want) {
            require(
                a.is_finite() && b.is_finite(),
                "nonfinite optimizer coordinate",
            )?;
            maximum = maximum.max(ulps(a, b));
        }
    }
    require(maximum <= 4, "GPU/FP32 reference exceeds four ulps")?;
    require(
        model::valid(Arm::Plain, &new.parameters) && model::valid(Arm::Plain, &new.first),
        "complex parameter/first-moment tie failed",
    )?;
    for [ar, bi, br, ai, kind] in model::pairs() {
        if kind == 0 {
            require(
                new.second[ar as usize] == new.second[ai as usize]
                    && new.second[bi as usize] == new.second[br as usize],
                "complex second-moment tie failed",
            )?;
        }
    }
    if completed == 0 {
        require(
            new.parameters == old.parameters,
            "zero-LR first step changed parameters",
        )?;
    }
    Ok(
        json!({"passed":true,"max_ulps":maximum,"norm_relative_error":norm_error,"norm":norm,"completed":completed}),
    )
}

pub fn preflight() -> Result<Value> {
    let data = Data::new(7);
    let plan = Plan::new(7, 10000);
    let p = model::initialize(Arm::Plain, 7);
    let mut e = Engine::new(Arm::Plain, &p, &data, &plan)?;
    let opt = Optimizer::new()?;
    let mut checks = vec![];
    for scale in [0.00001f32, 0.1] {
        let g: Vec<_> = (0..p.len())
            .map(|i| scale * ((i as f64 * 0.17).sin() as f32))
            .collect();
        #[cfg(not(feature = "settle"))]
        let steps = &[0, 1, 199, 200, 7999, 8000, 9000, 9999];
        #[cfg(feature = "settle")]
        let steps = &[
            0, 1, 199, 200, 999, 1000, 1001, 2500, 3999, 4000, 4001, 7999, 8000, 8001, 9000, 9999,
        ];
        for &completed in steps {
            let mut old = e.state()?;
            // Nonzero, unequal coordinate variances and tied memories.
            old.first = p.iter().map(|x| x * 0.03).collect();
            old.second = p.iter().map(|x| 0.001 + x * x).collect();
            e.restore(&old)?;
            e.gradients.write(&g)?;
            let norm = opt.step(&e, completed)?;
            checks.push(check(&old, &e.state()?, &g, norm, completed)?);
        }
    }
    Ok(json!({"passed":true,"cases":checks}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn scalar_update_agrees_with_real_adamw_and_uses_separate_variances() {
        let [p, m, v] = scalar(0.7, 0.02, 0.003, -0.4, 0.001, 0.19, 0.3122499);
        let mr = 0.9 * 0.02 + 0.1 * (-0.4);
        let vr: f64 = 0.95 * 0.003 + 0.05 * 0.16;
        let pr = 0.7 * (1. - 0.001 * 0.1) - (0.001 / 0.19) * mr / (vr.sqrt() / 0.3122499 + 1e-8);
        assert!((p as f64 - pr).abs() < 1e-7);
        assert!((m as f64 - mr).abs() < 1e-8);
        assert!((v as f64 - vr).abs() < 1e-8);
        let other = scalar(0.7, 0.02, 0.003, 0.1, 0.001, 0.19, 0.3122499);
        assert!(v > other[2]);
    }
    #[test]
    fn ownership_contraction_cancels_tied_null_gradients_before_norm() {
        let mut raw = vec![0.; Arm::Plain.count()];
        let [ar, bi, br, ai, _] = model::pairs().into_iter().find(|p| p[4] == 0).unwrap();
        raw[ar as usize] = 3.;
        raw[ai as usize] = -3.;
        raw[bi as usize] = 4.;
        raw[br as usize] = 4.;
        assert!(raw.iter().any(|&x| x != 0.));
        assert!(contracted(&raw).iter().all(|&x| x == 0.));
    }
}

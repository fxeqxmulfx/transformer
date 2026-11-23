//! Ordinary real-coordinate AdamW with the native Phasor cohort's WSD,
//! unit global norm clip, bias corrections and decoupled weight decay.
use crate::Result;
use tch::{Device, Kind, Tensor, nn};

#[kernite_macros::kernels]
mod kernels {
    pub unsafe extern "sysv64" fn adam(
        p: *mut f32,
        g: *const f32,
        m: *mut f32,
        v: *mut f32,
        c: *const f32,
        count: usize,
    ) {
        test(count, count);
        jz(done);
        let (norm, rate, bc2, decay, eps, a, b, u): (ymm, ymm, ymm, ymm, ymm, ymm, ymm, ymm);
        vbroadcastss(norm, [c]);
        vbroadcastss(rate, [c + 4]);
        vbroadcastss(bc2, [c + 8]);
        vbroadcastss(decay, [c + 12]);
        vbroadcastss(eps, [c + 16]);
        vbroadcastss(a, [c + 20]);
        vbroadcastss(b, [c + 24]);
        vbroadcastss(u, [c + 28]);
        label(next);
        let (grad, mom, var, tmp, w): (ymm, ymm, ymm, ymm, ymm);
        vmovups(grad, [g]);
        vdivps(grad, grad, norm);
        vmovups(mom, [m]);
        vsubps(tmp, grad, mom);
        vfmadd231ps(mom, a, tmp);
        vmovups([m], mom);
        vmulps(grad, grad, grad);
        vmulps(var, b, [v]);
        vfmadd231ps(var, u, grad);
        vmovups([v], var);
        vsqrtps(var, var);
        vdivps(var, var, bc2);
        vaddps(var, var, eps);
        vmulps(mom, mom, rate);
        vdivps(mom, mom, var);
        vmulps(w, decay, [p]);
        vsubps(w, w, mom);
        vmovups([p], w);
        add(p, 32);
        add(g, 32);
        add(m, 32);
        add(v, 32);
        dec(count);
        jnz(next);
        label(done);
    }
}

struct Parameter {
    value: Tensor,
    first: Tensor,
    second: Tensor,
}
pub struct Adam {
    rows: Vec<Parameter>,
}
impl Adam {
    pub fn new(vs: &nn::VarStore) -> Result<Self> {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            return Err("Kernite CPU AdamW requires AVX2 and FMA".into());
        }
        let mut variables: Vec<_> = vs.variables().into_iter().collect();
        variables.sort_by(|a, b| a.0.cmp(&b.0));
        let path = vs.root() / "optimizer";
        let mut rows = Vec::with_capacity(variables.len());
        for (index, (_, value)) in variables.into_iter().enumerate() {
            if value.device() != Device::Cpu
                || value.kind() != Kind::Float
                || !value.is_contiguous()
            {
                return Err("AdamW expects contiguous CPU f32 parameters".into());
            }
            let first = path.zeros_no_train(&format!("first{index}"), &value.size());
            let second = path.zeros_no_train(&format!("second{index}"), &value.size());
            rows.push(Parameter {
                value,
                first,
                second,
            });
        }
        Ok(Self { rows })
    }
    pub fn step(&mut self, rate: f32, completed: usize) -> Result<()> {
        let mut norm = 0f64;
        let mut gradients = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let g = row.value.grad();
            if !g.defined() {
                return Err("trainable parameter disconnected from loss".into());
            }
            let g = g.contiguous();
            let n = row.value.numel();
            if g.numel() != n {
                return Err("invalid gradient size".into());
            }
            // SAFETY: the contiguous CPU f32 tensor owns n floats and is retained below.
            let v = unsafe { std::slice::from_raw_parts(g.data_ptr().cast::<f32>(), n) };
            norm += v.iter().map(|&g| f64::from(g) * f64::from(g)).sum::<f64>();
            gradients.push(g);
        }
        if !norm.is_finite() {
            return Err("nonfinite training gradient".into());
        }
        let t = (completed + 1) as f64;
        let c = [
            (norm.sqrt() as f32).max(1.),
            rate / (1. - 0.9f64.powf(t)) as f32,
            (1. - 0.95f64.powf(t)).sqrt() as f32,
            1. - rate * 0.1,
            1e-8,
            0.1,
            0.95,
            0.05,
        ];
        for (row, g) in self.rows.iter_mut().zip(gradients) {
            let n = row.value.numel();
            let (p, m, v, g) = (
                row.value.data_ptr().cast::<f32>(),
                row.first.data_ptr().cast::<f32>(),
                row.second.data_ptr().cast::<f32>(),
                g.data_ptr().cast::<f32>(),
            );
            // SAFETY: the four distinct, live contiguous tensors own n f32s each.
            // The graph has finished backward. The kernel touches floor(n/8)*8 entries.
            unsafe {
                kernels::adam(p, g, m, v, c.as_ptr(), n / 8);
                for i in n / 8 * 8..n {
                    let grad = *g.add(i) / c[0];
                    let first = c[5].mul_add(grad - *m.add(i), *m.add(i));
                    let second = c[7].mul_add(grad * grad, c[6] * *v.add(i));
                    *m.add(i) = first;
                    *v.add(i) = second;
                    *p.add(i) = *p.add(i) * c[3] - c[1] * first / (second.sqrt() / c[2] + c[4]);
                }
            }
            row.value.zero_grad();
        }
        Ok(())
    }
}

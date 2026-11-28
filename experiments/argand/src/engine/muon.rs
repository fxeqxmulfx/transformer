//! Complex Muon on the attention and FFN projections. A shared batched
//! matrix product uses conjugate transposes. Tied embeddings, gates,
//! score scalars and frequencies use the paired AdamW auxiliary rule.
use super::{
    Engine, F, P, Result, adamw::RatePlan, grid, muon_kernels::device, page::Observations,
};
use crate::schedule::Schedule;
use kernite_cuda::{Buffer, Cuda, Grid};
use serde_json::{Value, json};
use std::collections::BTreeMap;

struct Matrices {
    rows: usize,
    cols: usize,
    count: usize,
    index: Buffer<u32>,
    moment: F,
    x: F,
    product: F,
    gram: F,
    polynomial: F,
}

impl Matrices {
    fn new(cuda: &Cuda, rows: usize, cols: usize, indices: &[u32]) -> Result<Self> {
        let n = indices.len();
        let count = n / (2 * rows * cols);
        let square = count * 2 * rows * rows;
        Ok(Self {
            rows,
            cols,
            count,
            index: cuda.upload(indices)?,
            moment: cuda.upload(&vec![0.0; n])?,
            x: cuda.alloc(n)?,
            product: cuda.alloc(n)?,
            gram: cuda.alloc(square)?,
            polynomial: cuda.alloc(square)?,
        })
    }

    /// The existing decoder GEMM takes row-major complex matrices in
    /// contiguous batches. Bits 1 and 4 of its flags transpose/conjugate B.
    fn multiply(
        &self,
        model: &Engine,
        dst: P,
        lhs: P,
        rhs: P,
        dims: [usize; 3],
        adjoint_rhs: bool,
    ) -> Result<()> {
        let [m, n, k] = dims;
        let tiles_m = m.div_ceil(16);
        let tiles_n = n.div_ceil(16);
        let flags = if adjoint_rhs { 2 | 16 } else { 0 };
        // SAFETY: the three shapes are supplied below from matching allocations.
        unsafe {
            model.kernel.matmul(
                Grid::new((tiles_m * tiles_n * self.count) as u32, 256),
                dst,
                lhs,
                rhs,
                m as u32,
                n as u32,
                k as u32,
                tiles_m as u32,
                tiles_n as u32,
                flags,
            )?;
        }
        Ok(())
    }

    fn step(&mut self, model: &Engine, kernel: &device::Kernels, norm: P, rate: f32) -> Result<()> {
        let n = self.index.len();
        let sq = self.gram.len();
        let (r, c) = (self.rows, self.cols);
        let p = &model.parameters;
        // SAFETY: maps contain unique in-range scalar coordinates; one warp normalizes a matrix.
        unsafe {
            kernel.momentum(
                grid(n),
                self.x.ptr(),
                self.moment.ptr(),
                p.grad.ptr(),
                self.index.ptr(),
                norm,
                n as u32,
            )?;
            kernel.normalize(
                Grid::new(self.count as u32, 32),
                self.x.ptr(),
                (2 * r * c) as u32,
            )?;
        }
        for _ in 0..5 {
            self.multiply(
                model,
                self.gram.ptr(),
                self.x.ptr(),
                self.x.ptr(),
                [r, r, c],
                true,
            )?;
            self.multiply(
                model,
                self.polynomial.ptr(),
                self.gram.ptr(),
                self.gram.ptr(),
                [r, r, r],
                false,
            )?;
            // SAFETY: polynomial and gram are equally sized batches of square matrices.
            unsafe {
                kernel.combine(
                    grid(sq),
                    self.polynomial.ptr(),
                    self.gram.ptr(),
                    sq as u32,
                    2.0315,
                    -4.775,
                )?;
            }
            self.multiply(
                model,
                self.product.ptr(),
                self.polynomial.ptr(),
                self.x.ptr(),
                [r, c, r],
                false,
            )?;
            // SAFETY: X and product have the original batch shape.
            unsafe {
                kernel.combine(
                    grid(n),
                    self.product.ptr(),
                    self.x.ptr(),
                    n as u32,
                    1.0,
                    3.4445,
                )?;
            }
            std::mem::swap(&mut self.x, &mut self.product);
        }
        // The group was packed with its shorter side first, so cols is its longer dimension.
        let scale = 0.2 * (c as f32).sqrt();
        unsafe {
            kernel.apply_matrix(
                grid(n),
                p.value.ptr(),
                self.x.ptr(),
                self.index.ptr(),
                n as u32,
                1.0 - rate * super::adamw::DECAY,
                rate * scale,
            )?;
        }
        Ok(())
    }
}

pub(crate) struct Muon {
    matrices: Vec<Matrices>,
    auxiliary: Buffer<u32>,
    first: F,
    second: F,
    kernels: device::Kernels,
    schedule: Schedule,
    scale: f64,
    completed: usize,
}

pub(crate) fn settings(scale: f64, schedule: Schedule) -> Value {
    json!({"peak":crate::schedule::PEAK*scale,"min_rate":1e-8*scale,"warmup":schedule.warmup,
        "cooldown_start":schedule.cooldown_start,"budget":schedule.budget,"weight_decay":0.1,"clip_norm":1,
        "momentum":0.95,"nesterov":true,"iterations":5,"coefficients":[3.4445,-4.775,2.0315],
        "frobenius_floor":1e-8,"matrix_scale":"0.2*sqrt(max(rows,cols))","matrices":"complex q,k,v,o,gate,up,down",
        "auxiliary":"tied embedding, gates, frequencies and scores: AdamW(0.9,0.95,1e-8)"})
}

impl Muon {
    pub fn new(model: &Engine, schedule: Schedule, scale: f64) -> Result<Self> {
        let mut matrix_indices = BTreeMap::<(usize, usize), Vec<u32>>::new();
        let mut used = vec![false; model.parameters.len()];
        for layer in &model.parameters.layers {
            for projection in layer.projections() {
                let (start, input, output) = projection.layout();
                let (r, c) = (input.min(output), input.max(output));
                let group = matrix_indices.entry((r, c)).or_default();
                for row in 0..r {
                    for col in 0..c {
                        let coordinate = if input <= output {
                            row * output + col
                        } else {
                            col * output + row
                        };
                        let offset = start + 2 * coordinate;
                        for j in offset..offset + 2 {
                            if std::mem::replace(&mut used[j], true) {
                                return Err("overlapping Muon projection map".into());
                            }
                            group.push(u32::try_from(j)?);
                        }
                    }
                }
            }
        }
        let matrices = matrix_indices
            .into_iter()
            .map(|((r, c), idx)| Matrices::new(&model.cuda, r, c, &idx))
            .collect::<Result<_>>()?;
        let auxiliary: Vec<_> = used
            .iter()
            .enumerate()
            .filter(|(_, used)| !**used)
            .map(|(i, _)| u32::try_from(i))
            .collect::<std::result::Result<_, _>>()?;
        let zeros = vec![0.0; auxiliary.len()];
        Ok(Self {
            matrices,
            auxiliary: model.cuda.upload(&auxiliary)?,
            first: model.cuda.upload(&zeros)?,
            second: model.cuda.upload(&zeros)?,
            kernels: device::Kernels::load(&model.cuda)?,
            schedule,
            scale,
            completed: 0,
        })
    }

    pub fn step(&mut self, model: &mut Engine, batch: Observations<'_>) -> Result<()> {
        batch.forward(model)?;
        model.backward()?;
        self.apply(model)
    }

    fn apply(&mut self, model: &Engine) -> Result<()> {
        let p = &model.parameters;
        let norm = p.reduce_norm(&model.optimizer, &p.grad)?.ptr();
        let rate = RatePlan::Scaled(self.scale).rate(self.schedule, self.completed);
        for matrices in &mut self.matrices {
            matrices.step(model, &self.kernels, norm, rate)?;
        }
        let t = (self.completed + 1) as f64;
        let n = self.auxiliary.len();
        // SAFETY: the auxiliary map is the disjoint complement of all projection maps.
        unsafe {
            self.kernels.auxiliary(
                grid(n),
                p.value.ptr(),
                self.first.ptr(),
                self.second.ptr(),
                p.grad.ptr(),
                self.auxiliary.ptr(),
                norm,
                n as u32,
                rate,
                1.0 - rate * super::adamw::DECAY,
                (1.0 - 0.9_f64.powf(t)) as f32,
                (1.0 - 0.95_f64.powf(t)).sqrt() as f32,
            )?;
        }
        self.completed += 1;
        Ok(())
    }

    pub fn metrics(&self) -> Value {
        json!({"step":self.completed,"rate":self.completed.checked_sub(1).map(|t| RatePlan::Scaled(self.scale).rate(self.schedule,t))})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires CUDA"]
    fn complex_rank_one_polynomial_tall_and_wide_and_auxiliary_adam() -> Result<()> {
        let cfg = crate::suite::Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 2,
        };
        let mut model = Engine::new(cfg, 1, 9)?;
        let schedule = super::super::adamw::wsd(20);
        let mut opt = Muon::new(&model, schedule, 1.0)?;
        let before = model.weights()?;
        let mut grad = vec![0.0; before.len()];
        grad[0] = 2.0;
        let group = &model.parameters.layers[0];
        let layouts = [group.q.layout(), group.up.layout(), group.down.layout()];
        let values = [0.3_f32, 0.4, -0.1, 0.2];
        for (at, _, _) in layouts {
            grad[at..at + 4].copy_from_slice(&values);
        }
        model.parameters.grad.write(&grad)?;
        opt.apply(&model)?;
        let after = model.weights()?;
        let eta = RatePlan::Wsd.rate(schedule, 0);
        assert!(eta > 0.0);
        let mut singular = 1.0_f64;
        for _ in 0..5 {
            singular = 3.4445 * singular - 4.775 * singular.powi(3) + 2.0315 * singular.powi(5);
        }
        let norm = values
            .iter()
            .map(|&x| f64::from(x).powi(2))
            .sum::<f64>()
            .sqrt();
        let mut expected: Vec<_> = before
            .iter()
            .map(|w| f64::from(*w * (1.0 - eta * 0.1)))
            .collect();
        expected[0] -= f64::from(eta);
        for (at, r, c) in layouts {
            for (j, &g) in values.iter().enumerate() {
                expected[at + j] -=
                    f64::from(eta) * 0.2 * (r.max(c) as f64).sqrt() * singular * f64::from(g)
                        / norm;
            }
        }
        for (x, y) in after.iter().zip(expected) {
            assert!((f64::from(*x) - y).abs() < 2e-7);
        }
        Ok(())
    }
}

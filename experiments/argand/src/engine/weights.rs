//! Independent real coordinates and clipped momentum SGD.
use super::kernels::device;
use super::tape::{Context, Tape};
use super::{F, Pair, Result, SCORE_INIT, grid};
use crate::{optimizer_kernels, random::Random, suite::Shape};
use kernite_cuda::{Cuda, Grid};

pub(super) struct Projection {
    at: usize,
    input: usize,
    output: usize,
}

impl Projection {
    pub(super) fn layout(&self) -> (usize, usize, usize) {
        (self.at, self.input, self.output)
    }

    fn new(weights: &mut Vec<f32>, rng: &mut Random, input: usize, output: usize) -> Self {
        let at = weights.len();
        weights.extend((0..2 * input * output).map(|_| rng.gauss(0.0, 0.02) as f32));
        Self { at, input, output }
    }

    pub fn project(
        &self,
        tape: &mut Tape,
        ctx: &Context<'_>,
        params: &Parameters,
        x: Pair,
        rows: usize,
    ) -> Result<Pair> {
        tape.mm(
            ctx,
            x.matrix(1, rows, self.input),
            params
                .pair(self.at, 2 * self.input * self.output)
                .matrix(1, self.input, self.output),
        )
    }
}

pub(super) struct Layer {
    pub q: Projection,
    pub k: Projection,
    pub v: Projection,
    pub o: Projection,
    pub gate: Projection,
    pub up: Projection,
    pub down: Projection,
    pub score: usize,
    /// The complex vectors that project the normalized attention and FFN
    /// inputs to their scalar write gates.
    pub write_gates: [usize; 2],
    /// The same for the skip gates.
    pub skip_gates: [usize; 2],
}

pub(super) struct Parameters {
    pub value: F,
    pub grad: F,
    velocity: F,
    parts: F,
    norm: F,
    embed_size: usize,
    frequencies: usize,
    pub layers: Vec<Layer>,
}

impl Parameters {
    pub fn new(cuda: &Cuda, cfg: &Shape, seed: u32) -> Result<Self> {
        let width = cfg.width;
        let mut rng = Random::new(seed);
        let embed_size = 2 * cfg.vocab * width;
        let frequencies = width / cfg.heads;
        let mut values: Vec<_> = (0..embed_size)
            .map(|_| rng.gauss(0.0, 0.02) as f32)
            .collect();
        values.resize(embed_size + frequencies, 0.0);
        let mut layers: Vec<_> = (0..cfg.layers)
            .map(|_| {
                let score = values.len();
                values.extend(SCORE_INIT);
                let mut projection =
                    |input, output| Projection::new(&mut values, &mut rng, input, output);
                Layer {
                    score,
                    q: projection(width, width),
                    k: projection(width, width),
                    v: projection(width, width),
                    o: projection(width, width),
                    gate: projection(width, 4 * width),
                    up: projection(width, 4 * width),
                    down: projection(4 * width, width),
                    write_gates: [0; 2],
                    skip_gates: [0; 2],
                }
            })
            .collect();
        // The checkpoint layout pads to an even count before the gates.
        values.resize(values.len().next_multiple_of(2), 0.0);
        for layer in &mut layers {
            let mut gate_vectors = || {
                let start = values.len();
                values.resize(start + 4 * width, 0.0);
                [start, start + 2 * width]
            };
            layer.write_gates = gate_vectors();
            layer.skip_gates = gate_vectors();
        }
        let count = values.len();
        Ok(Self {
            value: cuda.upload(&values)?,
            grad: cuda.alloc(count)?,
            velocity: cuda.upload(&vec![0.0; count])?,
            parts: cuda.alloc(count.div_ceil(256))?,
            norm: cuda.alloc(1)?,
            embed_size,
            frequencies,
            layers,
        })
    }

    /// The weights followed by the momentum velocity.
    pub fn state(&self) -> Result<Vec<f32>> {
        let mut values = self.value.read()?;
        values.extend(self.velocity.read()?);
        Ok(values)
    }

    pub fn restore_state(&mut self, state: &[f32]) -> Result<()> {
        let count = self.len();
        if state.len() != 2 * count || state.iter().any(|x| !x.is_finite()) {
            return Err("invalid decoder optimizer state".into());
        }
        self.value.write(&state[..count])?;
        self.velocity.write(&state[count..])?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    pub fn pair(&self, start: usize, len: usize) -> Pair {
        assert!(start + len <= self.value.len());
        Pair {
            x: self.value.at(start).unwrap(),
            dx: self.grad.at(start).unwrap(),
            len,
        }
    }

    pub fn embedding(&self) -> Pair {
        self.pair(0, self.embed_size)
    }
    pub fn alpha(&self) -> Pair {
        self.pair(self.embed_size, self.frequencies)
    }

    /// Adds `penalty * x` to the gradient, once and before `update` reads its
    /// norm, as `fullGradientObjective_gradient` adds the penalty before
    /// clipping.
    pub fn add_penalty(&self, device: &device::Kernels, penalty: f32) -> Result<()> {
        // SAFETY: the launch covers the equal value and gradient buffers.
        unsafe {
            device.add_penalty(
                grid(self.len()),
                self.grad.ptr(),
                self.value.ptr(),
                self.len() as u32,
                penalty,
            )?;
        }
        Ok(())
    }

    pub(super) fn reduce_norm(
        &self,
        kernel: &optimizer_kernels::kernels::Kernels,
        values: &F,
    ) -> Result<&F> {
        assert_eq!(values.len(), self.len());
        // SAFETY: each warp reduces one block of 256 coordinates.
        unsafe {
            kernel.norm_partial(
                Grid::new(self.parts.len() as u32, 32),
                self.parts.ptr(),
                values.ptr(),
                self.len() as u32,
            )?;
            kernel.norm_finish(
                Grid::new(1, 32),
                self.norm.ptr(),
                self.parts.ptr(),
                self.parts.len() as u32,
            )?;
        }
        Ok(&self.norm)
    }

    /// `v ← 0.9 v + g / max(‖g‖, 1)`, then `x ← x − rate · v`.
    pub fn update(
        &self,
        kernel: &optimizer_kernels::kernels::Kernels,
        device: &device::Kernels,
        rate: f32,
    ) -> Result<()> {
        self.reduce_norm(kernel, &self.grad)?;
        // SAFETY: each thread owns one coordinate in equal-sized buffers.
        unsafe {
            device.sgd_momentum(
                grid(self.len()),
                self.value.ptr(),
                self.velocity.ptr(),
                self.grad.ptr(),
                self.norm.ptr(),
                self.len() as u32,
                rate,
            )?;
        }
        Ok(())
    }

    pub fn check_sgd_momentum(
        &mut self,
        kernel: &optimizer_kernels::kernels::Kernels,
        device: &device::Kernels,
    ) -> Result<()> {
        let count = self.len();
        let initial = self.state()?;
        let mut expected = initial.clone();
        let mut grad = vec![0.0; count];
        // A clipped warmup step fills the velocity at zero rate. The next
        // steps retain it, including on coordinates with zero fresh gradient.
        for (completed, (rate, a, b)) in [
            (0.0_f32, 3.0_f32, 4.0_f32),
            (0.01, 0.1, 0.0),
            (0.02, -0.5, 0.1),
        ]
        .into_iter()
        .enumerate()
        {
            grad[0] = a;
            grad[1] = b;
            self.grad.write(&grad)?;
            self.update(kernel, device, rate)?;
            let scale = a.hypot(b).max(1.0);
            for (i, &raw) in grad.iter().enumerate() {
                expected[count + i] = 0.9_f32.mul_add(expected[count + i], raw / scale);
                expected[i] -= rate * expected[count + i];
            }
            let actual = self.state()?;
            if actual.len() != expected.len()
                || actual
                    .iter()
                    .zip(&expected)
                    .any(|(&x, &y)| !x.is_finite() || (x - y).abs() > 2e-6)
            {
                return Err(format!(
                    "SGD momentum update {} disagrees with the reference",
                    completed + 1
                )
                .into());
            }
        }
        let trained = self.state()?;
        self.restore_state(&initial)?;
        if self.state()? != initial {
            return Err("SGD momentum initial state restore failed".into());
        }
        self.restore_state(&trained)?;
        if self.state()? != trained {
            return Err("SGD momentum checkpoint restore failed".into());
        }
        println!(
            "SGD momentum clipping, zero-rate warmup, updates and checkpoint restore verified"
        );
        Ok(())
    }
}

//! Independent real coordinates and clipped Kernite AdamW and AMSGradW.
use super::complex_mhc::{AMPLITUDES, Routing, STREAMS};
use super::kernels::device;
use super::tape::{Context, Tape};
use super::{AmsGradWConfig, F, Pair, Result, Variant, grid};
use crate::{Config, optimizer_kernels, random::Random};
use kernite_cuda::{Buffer, Cuda, Grid};

pub(super) struct Projection {
    at: usize,
    input: usize,
    output: usize,
}

impl Projection {
    fn new(weights: &mut Vec<f32>, rng: &mut Random, input: usize, output: usize) -> Self {
        let at = weights.len();
        weights.extend((0..2 * input * output).map(|_| rng.gauss(0.0, 0.02) as f32));
        Self { at, input, output }
    }

    fn router(weights: &mut Vec<f32>, rng: &mut Random, input: usize, output: usize) -> Self {
        let at = weights.len();
        let deviation = 0.01 / (input as f64).sqrt();
        for _ in 0..input {
            weights.extend((0..AMPLITUDES).map(|_| rng.gauss(0.0, deviation) as f32));
            weights.resize(weights.len() + 2 * output - AMPLITUDES, 0.0);
        }
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
    pub routers: Option<[Projection; 2]>,
    /// Offsets `turn` and `turn + 1` hold the attention and FFN skip coefficients.
    pub turn: Option<usize>,
}

enum OptimizerBuffers {
    AdamW(Buffer<u32>),
    AmsGradW { maximum: F, config: AmsGradWConfig },
}

pub(super) struct Parameters {
    pub value: F,
    pub grad: F,
    first: F,
    second: F,
    optimizer: OptimizerBuffers,
    parts: F,
    norm: F,
    embed_size: usize,
    frequencies: usize,
    pub layers: Vec<Layer>,
}

impl Parameters {
    pub fn new(cuda: &Cuda, cfg: &Config, seed: u32, variant: Variant) -> Result<Self> {
        let width = cfg.width;
        let mut rng = Random::new(seed);
        let mut router_rng = Random::new(seed.wrapping_add(20_000));
        let embed_size = 2 * cfg.vocab * width;
        let frequencies = width / cfg.heads;
        let mut values: Vec<_> = (0..embed_size)
            .map(|_| rng.gauss(0.0, 0.02) as f32)
            .collect();
        values.resize(embed_size + frequencies, 0.0);
        let mut layers: Vec<_> = (0..cfg.layers)
            .map(|_| {
                let score = values.len();
                values.extend(variant.score_init());
                let mut projection =
                    |input, output| Projection::new(&mut values, &mut rng, input, output);
                Layer {
                    score,
                    turn: None,
                    q: projection(width, width),
                    k: projection(width, width),
                    v: projection(width, width),
                    o: projection(width, width),
                    gate: projection(width, 4 * width),
                    up: projection(width, 4 * width),
                    down: projection(4 * width, width),
                    routers: variant.routing.router_width().map(|output| {
                        std::array::from_fn(|_| {
                            Projection::router(
                                &mut values,
                                &mut router_rng,
                                STREAMS * width,
                                output,
                            )
                        })
                    }),
                }
            })
            .collect();
        // The legacy optimizer takes pairs. Pad one inert real coordinate if necessary.
        values.resize(values.len().next_multiple_of(2), 0.0);
        // Append after the entire ordinary model so every shared weight and its RNG draw match.
        if matches!(variant.routing, Routing::NormPhase) {
            for layer in &mut layers {
                layer.turn = Some(values.len());
                values.extend([0.0, 0.0]);
            }
        }
        let count = values.len();
        let zeros = vec![0.0; count];
        let optimizer = if let Some(config) = variant.amsgradw() {
            OptimizerBuffers::AmsGradW {
                maximum: cuda.upload(&zeros)?,
                config,
            }
        } else {
            let owners: Vec<[u32; 5]> = (0..count as u32)
                .step_by(2)
                .map(|i| [i, i + 1, 0, 0, 1])
                .collect();
            OptimizerBuffers::AdamW(cuda.upload(owners.as_flattened())?)
        };
        Ok(Self {
            value: cuda.upload(&values)?,
            grad: cuda.alloc(count)?,
            first: cuda.upload(&zeros)?,
            second: cuda.upload(&zeros)?,
            optimizer,
            parts: cuda.alloc(count.div_ceil(256))?,
            norm: cuda.alloc(1)?,
            embed_size,
            frequencies,
            layers,
        })
    }

    pub fn state(&self) -> Result<Vec<f32>> {
        let mut values = self.value.read()?;
        values.extend(self.first.read()?);
        values.extend(self.second.read()?);
        if let OptimizerBuffers::AmsGradW { maximum, .. } = &self.optimizer {
            values.extend(maximum.read()?);
        }
        Ok(values)
    }

    pub fn restore_state(&mut self, state: &[f32]) -> Result<()> {
        let count = self.len();
        let buffers = match &self.optimizer {
            OptimizerBuffers::AdamW(_) => 3,
            OptimizerBuffers::AmsGradW { .. } => 4,
        };
        if state.len() != buffers * count
            || state.iter().any(|x| !x.is_finite())
            || state[2 * count..].iter().any(|&x| x < 0.0)
        {
            return Err("invalid decoder optimizer state".into());
        }
        let second = &state[2 * count..3 * count];
        let peaks = &state[3 * count..];
        if peaks.iter().zip(second).any(|(peak, value)| peak < value) {
            return Err("AMSGradW maximum is below the second moment".into());
        }
        self.value.write(&state[..count])?;
        self.first.write(&state[count..2 * count])?;
        self.second.write(second)?;
        if let OptimizerBuffers::AmsGradW { maximum, .. } = &mut self.optimizer {
            maximum.write(peaks)?;
        }
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

    pub fn update(
        &self,
        kernel: &optimizer_kernels::kernels::Kernels,
        device: &device::Kernels,
        rate: f32,
        completed: usize,
    ) -> Result<()> {
        let grad = self.grad.ptr();
        let t = (completed + 1) as f64;
        let bc1 = (1.0 - 0.9_f64.powf(t)) as f32;
        let second_bias_correction = |beta2: f64| (1.0 - beta2.powf(t)).sqrt() as f32;
        // SAFETY: AdamW's identity map owns two coordinates per thread.
        // AMSGradW owns one coordinate per thread in all four buffers.
        unsafe {
            kernel.norm_partial(
                Grid::new(self.parts.len() as u32, 32),
                self.parts.ptr(),
                grad,
                self.len() as u32,
            )?;
            kernel.norm_finish(
                Grid::new(1, 32),
                self.norm.ptr(),
                self.parts.ptr(),
                self.parts.len() as u32,
            )?;
            match &self.optimizer {
                OptimizerBuffers::AdamW(map) => kernel.adam_coordinates(
                    grid(self.len() / 2),
                    self.value.ptr(),
                    self.first.ptr(),
                    self.second.ptr(),
                    grad,
                    map.ptr(),
                    self.norm.ptr(),
                    (self.len() / 2) as u32,
                    rate,
                    0.1,
                    bc1,
                    second_bias_correction(0.95),
                )?,
                OptimizerBuffers::AmsGradW { maximum, config } => device.amsgradw(
                    grid(self.len()),
                    self.value.ptr(),
                    self.first.ptr(),
                    self.second.ptr(),
                    maximum.ptr(),
                    grad,
                    self.norm.ptr(),
                    self.len() as u32,
                    rate,
                    0.1,
                    config.beta2 as f32,
                    (1.0 - config.beta2) as f32,
                    bc1,
                    second_bias_correction(config.beta2),
                    config.sqrt_scale,
                )?,
            }
        }
        Ok(())
    }

    pub fn check_amsgradw(
        &mut self,
        kernel: &optimizer_kernels::kernels::Kernels,
        device: &device::Kernels,
        config: AmsGradWConfig,
    ) -> Result<()> {
        let count = self.len();
        let mut grad = vec![0.0; count];
        let mut value = f64::from(self.value.read()?[0]);
        let (mut first, mut second, mut maximum) = (0.0_f64, 0.0_f64, 0.0_f64);
        let rate = 0.01_f32;
        let lr = f64::from(rate);
        // The first gradient clips and the next two exercise the running maximum.
        for (completed, raw) in [2.0_f32, 0.1, -0.5].into_iter().enumerate() {
            grad[0] = raw;
            self.grad.write(&grad)?;
            self.update(kernel, device, rate, completed)?;
            let g = f64::from(raw / raw.abs().max(1.0));
            first = 0.9 * first + 0.1 * g;
            second = config.beta2 * second + (1.0 - config.beta2) * g * g;
            maximum = maximum.max(second);
            let step = (completed + 1) as i32;
            let corrected_first = first / (1.0 - 0.9_f64.powi(step));
            let corrected_maximum = maximum / (1.0 - config.beta2.powi(step));
            let denominator = f64::from(config.sqrt_scale) * corrected_maximum.sqrt() + 1e-8;
            value = (1.0 - lr * 0.1) * value - lr * corrected_first / denominator;
            let state = self.state()?;
            for (buffer, expected) in [value, first, second, maximum].into_iter().enumerate() {
                let actual = f64::from(state[buffer * count]);
                if !actual.is_finite() || (actual - expected).abs() > 2e-6 {
                    return Err(format!(
                        "AMSGradW step {step}, buffer {buffer}: expected {expected}, got {actual}"
                    )
                    .into());
                }
            }
        }
        println!("AMSGradW clipping, moments, maximum and parameter updates verified");
        Ok(())
    }
}

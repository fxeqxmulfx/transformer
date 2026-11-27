//! Independent real coordinates and the existing clipped Kernite AdamW kernel.
use super::complex_mhc::{AMPLITUDES, STREAMS};
use super::tape::{Context, Tape};
use super::{F, Pair, Result, Variant, grid};
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
}

pub(super) struct Parameters {
    pub value: F,
    pub grad: F,
    first: F,
    second: F,
    map: Buffer<u32>,
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
        let layers = (0..cfg.layers)
            .map(|_| {
                let score = values.len();
                values.extend(variant.score_init());
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
        let count = values.len();
        let owners: Vec<[u32; 5]> = (0..count as u32)
            .step_by(2)
            .map(|i| [i, i + 1, 0, 0, 1])
            .collect();
        let zeros = vec![0.0; count];
        Ok(Self {
            value: cuda.upload(&values)?,
            grad: cuda.alloc(count)?,
            first: cuda.upload(&zeros)?,
            second: cuda.upload(&zeros)?,
            map: cuda.upload(owners.as_flattened())?,
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
        Ok(values)
    }

    pub fn restore_state(&mut self, state: &[f32]) -> Result<()> {
        let count = self.len();
        if state.len() != 3 * count
            || state.iter().any(|x| !x.is_finite())
            || state[2 * count..].iter().any(|&x| x < 0.0)
        {
            return Err("invalid Phasor v2 optimizer state".into());
        }
        self.value.write(&state[..count])?;
        self.first.write(&state[count..2 * count])?;
        self.second.write(&state[2 * count..])?;
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
        rate: f32,
        completed: usize,
    ) -> Result<()> {
        let grad = self.grad.ptr();
        let t = (completed + 1) as f64;
        // SAFETY: each pair in the identity map owns two distinct coordinates.
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
            kernel.adam_coordinates(
                grid(self.len() / 2),
                self.value.ptr(),
                self.first.ptr(),
                self.second.ptr(),
                grad,
                self.map.ptr(),
                self.norm.ptr(),
                (self.len() / 2) as u32,
                rate,
                0.1,
                (1.0 - 0.9_f64.powf(t)) as f32,
                (1.0 - 0.95_f64.powf(t)).sqrt() as f32,
            )?;
        }
        Ok(())
    }
}

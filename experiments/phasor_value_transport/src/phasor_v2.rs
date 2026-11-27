//! Argand's complex decoder and tied vocabulary readout on Kernite.
mod complex_mhc;
mod kernels;
mod scores;
mod tape;
mod weights;

pub(crate) use crate::Result;
use crate::{base_ops, optimizer_kernels, random::Random, slot_copy, stable_loss};
use complex_mhc::{Routing, Sublayer};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
use tape::{Context, Pair, Tape};
use weights::Parameters;

type F = Buffer<f32>;
type P = Ptr<f32>;

#[derive(Clone, Copy)]
pub struct Variant {
    routing: Routing,
    real_score_init: bool,
    amsgradw: bool,
}

impl Variant {
    pub fn argand() -> Result<Self> {
        let real_score_init = match std::env::var("ARGAND_SCORE_INIT").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("1+i") => false,
            Ok("1") => true,
            _ => return Err("ARGAND_SCORE_INIT takes 1+i or 1".into()),
        };
        let amsgradw = match std::env::var("ARGAND_OPTIMIZER").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("adamw") => false,
            Ok("amsgradw") => true,
            _ => return Err("ARGAND_OPTIMIZER takes adamw or amsgradw".into()),
        };
        Ok(Self {
            routing: Routing::EntryPhase,
            real_score_init,
            amsgradw,
        })
    }

    pub fn v2() -> Result<Self> {
        Ok(Self {
            routing: Routing::selected()?,
            real_score_init: false,
            amsgradw: false,
        })
    }

    pub fn model(self) -> &'static str {
        match (self.real_score_init, self.amsgradw) {
            (false, false) => self.routing.model(),
            (true, false) => "argand-entry-phase-mhc-tied-vocab-fp32-s-init-one",
            (false, true) => "argand-entry-phase-mhc-tied-vocab-fp32-amsgradw",
            (true, true) => "argand-entry-phase-mhc-tied-vocab-fp32-s-init-one-amsgradw",
        }
    }

    pub fn legacy_model(self) -> Option<&'static str> {
        // Old entry-phase checkpoints used AdamW and started the score at 1+i.
        (matches!(self.routing, Routing::EntryPhase) && !self.real_score_init && !self.amsgradw)
            .then_some("phasor-v2-entry-phase-mhc-tied-vocab-fp32")
    }

    fn score_init(self) -> [f32; 2] {
        [1.0, if self.real_score_init { 0.0 } else { 1.0 }]
    }
}

fn grid(count: usize) -> Grid {
    Grid::new(count.div_ceil(256) as u32, 256)
}

pub struct Scoring {
    pub max_batch: usize,
    pub vocab: usize,
}

pub struct Engine {
    pub config: Scoring,
    pub predictions: Buffer<u32>,
    pub logits: F,
    pub cuda: Cuda,
    shape: crate::Config,
    repeat: usize,
    parameters: Parameters,
    tape: Tape,
    kernel: kernels::device::Kernels,
    scores: scores::device::Kernels,
    mhc: Option<complex_mhc::Kernels>,
    embed: base_ops::kernels::Kernels,
    loss: stable_loss::kernels::Kernels,
    clear: slot_copy::kernels::Kernels,
    adam: optimizer_kernels::kernels::Kernels,
    inputs: Buffer<u32>,
    labels: Buffer<u32>,
    loss_values: F,
    logit_grad: F,
    rotations: F,
    last: Option<(Pair, Pair, usize, usize)>,
}

impl Engine {
    pub fn new(shape: crate::Config, repeat: usize, seed: u32, variant: Variant) -> Result<Self> {
        if shape.width == 0
            || shape.heads == 0
            || !shape.width.is_multiple_of(shape.heads)
            || shape.layers == 0
            || repeat == 0
            || shape.layers.checked_mul(repeat).is_none()
        {
            return Err("invalid Phasor v2 model dimensions".into());
        }
        let cuda = Cuda::new()?;
        let rows = shape.max_batch * shape.max_length;
        let vocab = shape.vocab;
        let logits = rows * vocab;
        let config = Scoring {
            max_batch: shape.max_batch,
            vocab,
        };
        Ok(Self {
            config,
            repeat,
            parameters: Parameters::new(&cuda, &shape, seed, variant)?,
            tape: Tape::default(),
            kernel: kernels::device::Kernels::load(&cuda)?,
            scores: scores::device::Kernels::load(&cuda)?,
            mhc: variant
                .routing
                .router_width()
                .map(|_| complex_mhc::Kernels::load(&cuda))
                .transpose()?,
            embed: base_ops::kernels::Kernels::load(&cuda)?,
            loss: stable_loss::kernels::Kernels::load(&cuda)?,
            clear: slot_copy::kernels::Kernels::load(&cuda)?,
            adam: optimizer_kernels::kernels::Kernels::load(&cuda)?,
            inputs: cuda.alloc(rows)?,
            labels: cuda.alloc(rows)?,
            predictions: cuda.alloc(rows)?,
            logits: cuda.alloc(logits)?,
            logit_grad: cuda.alloc(logits)?,
            loss_values: cuda.alloc(rows)?,
            rotations: cuda.alloc(3 * shape.max_length * (shape.width / shape.heads))?,
            last: None,
            shape,
            cuda,
        })
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<()> {
        let rows = batch * length;
        if batch == 0
            || batch > self.shape.max_batch
            || length == 0
            || length > self.shape.max_length
            || tokens.len() != rows
            || labels.len() != rows
            || tokens.iter().any(|&id| id as usize >= self.config.vocab)
        {
            return Err("invalid Phasor v2 input batch".into());
        }
        let labels_count = labels
            .iter()
            .filter(|&&y| (y as usize) < self.config.vocab)
            .count();
        self.last = None;
        self.inputs
            .part_mut(..rows)
            .ok_or("input capacity")?
            .write(tokens)?;
        self.labels
            .part_mut(..rows)
            .ok_or("label capacity")?
            .write(labels)?;
        self.tape.reset();
        let width = self.shape.width;
        let input = self.tape.allocate(&self.cuda, rows * 2 * width)?;
        // SAFETY: all input IDs and lengths were validated above.
        unsafe {
            self.embed.embed(
                grid(input.len),
                input.x,
                self.parameters.embedding().x,
                self.inputs.ptr(),
                (2 * width) as u32,
                input.len as u32,
            )?;
        }
        let hidden = self.inner(input, batch, length)?;
        let scores = self.readout(hidden, batch, length)?;
        // SAFETY: stable loss receives one complete vocabulary row per position.
        unsafe {
            self.loss.loss(
                Grid::new(rows.div_ceil(8) as u32, 256),
                self.logit_grad.ptr(),
                self.loss_values.ptr(),
                self.predictions.ptr(),
                self.logits.ptr(),
                self.labels.ptr(),
                self.config.vocab as u32,
                rows as u32,
                labels_count.max(1) as f32,
            )?;
        }
        self.last = Some((input, scores, batch, length));
        Ok(())
    }

    fn inner(&mut self, input: Pair, batch: usize, length: usize) -> Result<Pair> {
        let rows = batch * length;
        let p = &self.parameters;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let head_width = width / heads;
        let alpha = p.alpha();
        assert_eq!(input.len, rows * 2 * width);
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            scores: &self.scores,
            mhc: self.mhc.as_ref(),
            rotations: self.rotations.ptr(),
            alpha,
            length,
            heads,
            head_width,
        };
        let tape = &mut self.tape;
        // SAFETY: the table has three reals per frequency and position.
        unsafe {
            self.kernel.rope_table(
                grid(length * head_width),
                ctx.rotations,
                alpha.x,
                head_width as u32,
                (length * head_width) as u32,
                -10_000_f32.ln() / head_width as f32,
            )?;
        }
        let mut x = tape.rms(&ctx, input, width)?;
        if self.mhc.is_some() {
            x = tape.expand(&ctx, x, width)?;
        }
        let scale = (2.0 * (self.shape.layers * self.repeat) as f32)
            .sqrt()
            .recip();
        for _ in 0..self.repeat {
            for layer in &p.layers {
                let router = layer.routers.as_ref();
                let attention =
                    Sublayer::read(tape, &ctx, p, router.map(|r| &r[0]), x, rows, width)?;
                let n = tape.rms(&ctx, attention.input, width)?;
                let q = layer.q.project(tape, &ctx, p, n, rows)?;
                let k = layer.k.project(tape, &ctx, p, n, rows)?;
                let v = layer.v.project(tape, &ctx, p, n, rows)?;
                let q = tape.rms(&ctx, q, head_width)?;
                let k = tape.rms(&ctx, k, head_width)?;
                let q = tape.rotate(&ctx, q)?;
                let k = tape.rotate(&ctx, k)?;
                let q = tape
                    .heads(&ctx, q, false)?
                    .matrix(batch * heads, length, head_width);
                let k = tape
                    .heads(&ctx, k, false)?
                    .matrix(batch * heads, length, head_width);
                let v = tape
                    .heads(&ctx, v, false)?
                    .matrix(batch * heads, length, head_width);
                let score = tape.mm(&ctx, q.conjugate(), k.transpose())?;
                let score = tape.scores(&ctx, score, p.pair(layer.score, 2))?;
                let a = tape.mm(&ctx, score.matrix(batch * heads, length, length), v)?;
                let a = tape.heads(&ctx, a, true)?;
                let a = layer.o.project(tape, &ctx, p, a, rows)?;
                let a = tape.rms(&ctx, a, width)?;
                x = attention.write(tape, &ctx, a, width, scale)?;

                let ffn = Sublayer::read(tape, &ctx, p, router.map(|r| &r[1]), x, rows, width)?;
                let n = tape.rms(&ctx, ffn.input, width)?;
                let g = layer.gate.project(tape, &ctx, p, n, rows)?;
                let u = layer.up.project(tape, &ctx, p, n, rows)?;
                let f = tape.product(&ctx, g, u)?;
                let f = layer.down.project(tape, &ctx, p, f, rows)?;
                let f = tape.rms(&ctx, f, width)?;
                x = ffn.write(tape, &ctx, f, width, scale)?;
            }
        }
        if self.mhc.is_some() {
            x = tape.collapse(&ctx, x, width)?;
        }
        tape.rms(&ctx, x, width)
    }

    fn readout(&mut self, hidden: Pair, batch: usize, length: usize) -> Result<Pair> {
        let rows = batch * length;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            scores: &self.scores,
            mhc: self.mhc.as_ref(),
            rotations: self.rotations.ptr(),
            alpha: self.parameters.alpha(),
            length,
            heads,
            head_width: width / heads,
        };
        let h = hidden.matrix(1, rows, width);
        let classifier = self
            .parameters
            .embedding()
            .matrix(1, self.config.vocab, width)
            .transpose()
            .conjugate();
        let scores = self.tape.mm(&ctx, h, classifier)?;
        let count = rows * self.config.vocab;
        // SAFETY: each row of scores covers the entire vocabulary.
        unsafe {
            self.kernel.real_readout::<false>(
                grid(count),
                self.logits.ptr(),
                scores.x,
                count as u32,
            )?;
        }
        Ok(scores)
    }

    pub fn losses(&self, rows: usize) -> Result<Vec<f32>> {
        Ok(self
            .loss_values
            .part(..rows)
            .ok_or("loss capacity")?
            .read()?)
    }

    fn backward(&mut self) -> Result<()> {
        let (input, output, batch, length) =
            self.last.take().ok_or("forward required before update")?;
        let width = self.shape.width;
        let heads = self.shape.heads;
        let ctx = Context {
            cuda: &self.cuda,
            kernel: &self.kernel,
            scores: &self.scores,
            mhc: self.mhc.as_ref(),
            rotations: self.rotations.ptr(),
            alpha: self.parameters.alpha(),
            length,
            heads,
            head_width: width / heads,
        };
        let p = &self.parameters;
        self.tape.clear_gradients(&self.clear)?;
        // SAFETY: these launches use the parameter and logit buffer extents.
        unsafe {
            self.clear
                .clear(grid(p.len()), p.grad.ptr(), p.len() as u32)?;
            let count = output.len / 2;
            self.kernel.real_readout::<true>(
                grid(count),
                output.dx,
                self.logit_grad.ptr(),
                count as u32,
            )?;
        }
        self.tape.backward(&ctx)?;
        // SAFETY: input IDs were checked and this gradient adds to the tied readout's.
        unsafe {
            let e = p.embedding();
            self.embed.embed_bwd(
                grid(e.len),
                e.dx,
                input.dx,
                self.inputs.ptr(),
                (2 * width) as u32,
                self.config.vocab as u32,
                (batch * length) as u32,
            )?;
        }
        Ok(())
    }

    pub fn step(&mut self, rate: f32, completed: usize) -> Result<()> {
        self.backward()?;
        self.parameters
            .update(&self.adam, &self.kernel, rate, completed)
    }

    pub fn state(&self) -> Result<Vec<f32>> {
        self.parameters.state()
    }

    pub fn restore_state(&mut self, state: &[f32]) -> Result<()> {
        self.last = None;
        self.parameters.restore_state(state)
    }

    pub fn weights(&self) -> Result<Vec<f32>> {
        Ok(self.parameters.value.read()?)
    }

    pub fn restore(&mut self, weights: &[f32]) -> Result<()> {
        if weights.len() != self.parameters.len() || weights.iter().any(|x| !x.is_finite()) {
            return Err("invalid Phasor v2 checkpoint".into());
        }
        self.last = None;
        Ok(self.parameters.value.write(weights)?)
    }
}

pub fn check(variant: Variant) -> Result<()> {
    let mut cfg = crate::config(2)?;
    cfg.width = 16;
    cfg.heads = 2;
    cfg.layers = 2;
    cfg.vocab = 6;
    cfg.max_length = 3;
    let mut engine = Engine::new(cfg, 2, 1, variant)?;
    let initial = engine.weights()?;
    if engine
        .parameters
        .layers
        .iter()
        .any(|layer| initial[layer.score..layer.score + 2] != variant.score_init())
    {
        return Err("score multiplier has the wrong initialization".into());
    }
    let mut tied = vec![0.0; initial.len()];
    tied[2 * 16] = 0.125;
    tied[2 * 2 * 16] = 0.25;
    engine.restore(&tied)?;
    engine.forward(&[1], &[2], 1, 1)?;
    let logits = engine.logits.part(..6).ok_or("logit capacity")?.read()?;
    let loss = engine.losses(1)?[0];
    if engine.predictions.read()?[0] != 2
        || !loss.is_finite()
        || (logits[2] - 0.5).abs() > 1e-5
        || (logits[1] - 0.25).abs() > 1e-5
    {
        return Err("the tied classifier failed to predict the unseen token".into());
    }

    engine.restore(&initial)?;
    let tokens = [1, 2, 3, 3, 1, 0];
    let labels = [2, 3, 4, u32::MAX, 5, 2];
    engine.forward(&tokens, &labels, 2, 3)?;
    let before = engine.logits.read()?;
    let dense: Vec<_> = tokens
        .iter()
        .flat_map(|&id| {
            initial[id as usize * 32..(id as usize + 1) * 32]
                .iter()
                .copied()
        })
        .collect();
    let dense_input = engine.cuda.upload(&dense)?;
    let dense_grad = engine.cuda.alloc(dense.len())?;
    engine.tape.reset();
    let hidden = engine.inner(
        Pair {
            x: dense_input.ptr(),
            dx: dense_grad.ptr(),
            len: dense.len(),
        },
        2,
        3,
    )?;
    engine.readout(hidden, 2, 3)?;
    if engine.logits.read()? != before {
        return Err("dense embedding input disagrees with the token lookup".into());
    }
    engine.forward(&[1, 2, 5, 3, 1, 5], &[0; 6], 2, 3)?;
    let after = engine.logits.read()?;
    for i in (0..12).chain(18..30) {
        if (before[i] - after[i]).abs() > 2e-6 {
            return Err("future inputs or labels changed a prefix logit".into());
        }
    }

    engine.forward(&tokens, &labels, 2, 3)?;
    engine.backward()?;
    let gradient = engine.parameters.grad.read()?;
    if !gradient.iter().all(|g| g.is_finite())
        || !gradient[2 * 4 * 16..2 * 5 * 16]
            .iter()
            .any(|g| g.abs() > 1e-6)
    {
        return Err("missing gradient for a vocabulary row absent from lookup".into());
    }
    if variant.real_score_init
        && !engine
            .parameters
            .layers
            .iter()
            .any(|layer| gradient[layer.score + 1].abs() > 1e-6)
    {
        return Err("missing gradient for the imaginary part of the score multiplier".into());
    }
    let mut rng = Random::new(100);
    for trial in 0..3 {
        let direction: Vec<_> = initial.iter().map(|_| rng.gauss(0.0, 1.0) as f32).collect();
        let delta = 0.0002;
        let mut losses = [0.0; 2];
        for (loss, sign) in losses.iter_mut().zip([-1.0, 1.0]) {
            let weights: Vec<_> = initial
                .iter()
                .zip(&direction)
                .map(|(x, d)| x + d * delta * sign)
                .collect();
            engine.restore(&weights)?;
            engine.forward(&tokens, &labels, 2, 3)?;
            *loss = engine.losses(6)?.iter().map(|&x| f64::from(x)).sum::<f64>() / 5.0;
        }
        let finite = (losses[1] - losses[0]) / f64::from(2.0 * delta);
        let analytic: f64 = gradient
            .iter()
            .zip(&direction)
            .map(|(&g, &d)| f64::from(g) * f64::from(d))
            .sum();
        println!("direction {trial}: analytic={analytic:.6} finite_difference={finite:.6}");
        if !finite.is_finite() || (finite - analytic).abs() > 0.02 * (1.0 + finite.abs()) {
            return Err("complex decoder GPU backward disagrees with finite differences".into());
        }
    }
    if variant.amsgradw {
        engine
            .parameters
            .check_amsgradw(&engine.adam, &engine.kernel)?;
    }
    println!(
        "{} vocabulary prediction, prefix causality and tied adjoints verified",
        variant.model()
    );
    Ok(())
}

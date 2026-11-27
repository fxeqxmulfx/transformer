//! Argand's complex decoder and tied vocabulary readout on Kernite.
mod complex_mhc;
mod kernels;
mod norm_phase_checks;
mod scores;
mod tape;
mod weights;

pub(crate) use crate::Result;
use crate::{base_ops, optimizer_kernels, random::Random, slot_copy, stable_loss};
use complex_mhc::{Routing, Sublayer};
use kernite_cuda::{Buffer, Cuda, Grid, Ptr};
use tape::{Context, Pair, Tape};
use weights::{Parameters, SecondGate};

type F = Buffer<f32>;
type P = Ptr<f32>;

#[derive(Clone, Copy)]
pub(crate) struct AmsGradWConfig {
    pub beta2: f64,
    pub sqrt_scale: f32,
}

impl AmsGradWConfig {
    pub const DEFAULT_BETA2: f64 = 0.95;
}

#[derive(Clone, Copy)]
enum Optimizer {
    SgdMomentum,
    AdamW,
    AmsGradW(AmsGradWConfig),
}

#[derive(Clone, Copy)]
enum Gradient {
    Minibatch,
    /// Every update follows the Lean `fullGradientMomentumRun`: the gradient of
    /// the mean loss over all labeled training positions plus `penalty * x`.
    Full {
        penalty: f32,
    },
}

#[derive(Clone, Copy)]
pub struct Variant {
    routing: Routing,
    real_score_init: bool,
    optimizer: Optimizer,
    gradient: Gradient,
}

impl Variant {
    pub fn argand() -> Result<Self> {
        let routing = match std::env::var("ARGAND_RESIDUAL").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("skip-write-gate") => Routing::SkipWriteGate,
            Ok("entry-phase") => Routing::EntryPhase,
            Ok("single") => Routing::Single,
            Ok("norm-phase") => Routing::NormPhase,
            Ok("write-gate") => Routing::WriteGate,
            Ok("read-write-gate") => Routing::ReadWriteGate,
            _ => {
                return Err(
                    "ARGAND_RESIDUAL takes entry-phase, single, norm-phase, write-gate, read-write-gate or skip-write-gate".into(),
                );
            }
        };
        let real_score_init = match std::env::var("ARGAND_SCORE_INIT").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("1+i") => false,
            Ok("1") => true,
            _ => return Err("ARGAND_SCORE_INIT takes 1+i or 1".into()),
        };
        let optimizer = match std::env::var("ARGAND_OPTIMIZER").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("sgd-momentum") => Optimizer::SgdMomentum,
            Ok("adamw") => Optimizer::AdamW,
            Ok("amsgradw") => {
                let sqrt_scale = match std::env::var("ARGAND_AMSGRADW_SQRT_SCALE") {
                    Ok(value) => value.parse::<f32>()?,
                    Err(std::env::VarError::NotPresent) => 1.0,
                    Err(error) => return Err(error.into()),
                };
                if !(0.0 < sqrt_scale && sqrt_scale <= 1.0) {
                    return Err("ARGAND_AMSGRADW_SQRT_SCALE must be in (0, 1]".into());
                }
                let beta2 = match std::env::var("ARGAND_AMSGRADW_BETA2") {
                    Ok(value) => value.parse::<f64>()?,
                    Err(std::env::VarError::NotPresent) => AmsGradWConfig::DEFAULT_BETA2,
                    Err(error) => return Err(error.into()),
                };
                if !(0.0..1.0).contains(&beta2) {
                    return Err("ARGAND_AMSGRADW_BETA2 must be in [0, 1)".into());
                }
                Optimizer::AmsGradW(AmsGradWConfig { beta2, sqrt_scale })
            }
            _ => return Err("ARGAND_OPTIMIZER takes sgd-momentum, adamw or amsgradw".into()),
        };
        let penalty = std::env::var("ARGAND_PENALTY");
        let gradient = match std::env::var("ARGAND_BATCH").as_deref() {
            Err(std::env::VarError::NotPresent) | Ok("minibatch") => {
                if !matches!(penalty, Err(std::env::VarError::NotPresent)) {
                    return Err("ARGAND_PENALTY requires ARGAND_BATCH=full".into());
                }
                Gradient::Minibatch
            }
            Ok("full") => {
                let penalty = penalty
                    .map_err(|_| "ARGAND_BATCH=full requires ARGAND_PENALTY, the coefficient of (λ/2)‖x‖²")?
                    .parse::<f32>()
                    .map_err(|_| "ARGAND_PENALTY must be a number")?;
                if !(penalty.is_finite() && penalty > 0.0) {
                    return Err("ARGAND_PENALTY must be positive and finite".into());
                }
                if !matches!(optimizer, Optimizer::SgdMomentum) {
                    return Err("ARGAND_BATCH=full trains with clipped momentum SGD; leave ARGAND_OPTIMIZER unset or set it to sgd-momentum".into());
                }
                if !matches!(routing, Routing::SkipWriteGate) {
                    return Err("ARGAND_BATCH=full trains the Lean Argand model with skip and write gates; leave ARGAND_RESIDUAL unset or set it to skip-write-gate".into());
                }
                Gradient::Full { penalty }
            }
            _ => return Err("ARGAND_BATCH takes minibatch or full".into()),
        };
        Ok(Self {
            routing,
            real_score_init,
            optimizer,
            gradient,
        })
    }

    pub fn v2() -> Result<Self> {
        Ok(Self {
            routing: Routing::selected()?,
            real_score_init: false,
            optimizer: Optimizer::AdamW,
            gradient: Gradient::Minibatch,
        })
    }

    pub fn model(self) -> String {
        let mut name = self.routing.model().to_owned();
        if self.real_score_init {
            name.push_str("-s-init-one");
        }
        match self.optimizer {
            Optimizer::SgdMomentum => name.push_str("-sgd-momentum"),
            Optimizer::AdamW => {}
            Optimizer::AmsGradW(_) => name.push_str("-amsgradw"),
        }
        if let Gradient::Full { .. } = self.gradient {
            name.push_str("-full-batch");
        }
        name
    }

    /// The coefficient λ of the full-gradient penalty `(λ/2)‖x‖²`.
    pub(crate) fn full_batch_penalty(self) -> Option<f32> {
        match self.gradient {
            Gradient::Minibatch => None,
            Gradient::Full { penalty } => Some(penalty),
        }
    }

    pub fn legacy_model(self) -> Option<&'static str> {
        // Old entry-phase checkpoints used AdamW and started the score at 1+i.
        (matches!(self.routing, Routing::EntryPhase)
            && !self.real_score_init
            && matches!(self.optimizer, Optimizer::AdamW))
        .then_some("phasor-v2-entry-phase-mhc-tied-vocab-fp32")
    }

    pub(crate) fn amsgradw(self) -> Option<AmsGradWConfig> {
        match self.optimizer {
            Optimizer::AmsGradW(config) => Some(config),
            _ => None,
        }
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
            return Err("invalid Argand model dimensions".into());
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

    /// The number of supervised positions: labels below the vocabulary size.
    pub fn labeled(&self, labels: &[u32]) -> usize {
        labels
            .iter()
            .filter(|&&y| (y as usize) < self.config.vocab)
            .count()
    }

    pub fn forward(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
    ) -> Result<()> {
        self.forward_normalized(tokens, labels, batch, length, None)
    }

    /// The loss gradient divides by `normalizer`, by default this batch's
    /// own label count.
    fn forward_normalized(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
        normalizer: Option<usize>,
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
            return Err("invalid Argand input batch".into());
        }
        let own = self.labeled(labels).max(1);
        let normalizer = match normalizer {
            None => own,
            Some(total) if total >= own => total,
            Some(_) => return Err("the full-batch label count is below a microbatch's".into()),
        };
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
                normalizer as f32,
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
                let attention = Sublayer::read(tape, &ctx, p, layer, 0, x, rows, width)?;
                let n = attention.input;
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

                let ffn = Sublayer::read(tape, &ctx, p, layer, 1, x, rows, width)?;
                let n = ffn.input;
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
        self.zero_gradient()?;
        self.add_backward()
    }

    /// Clears the parameter gradient before a full-batch accumulation.
    pub fn zero_gradient(&mut self) -> Result<()> {
        let p = &self.parameters;
        // SAFETY: the launch covers the parameter gradient buffer.
        unsafe {
            self.clear
                .clear(grid(p.len()), p.grad.ptr(), p.len() as u32)?;
        }
        Ok(())
    }

    /// Adds one microbatch's part of the mean gradient over `total` labeled
    /// positions. Over a partition of the training split the parts sum to
    /// the gradient of the mean loss (`accumulated_batchGradient`).
    pub fn accumulate(
        &mut self,
        tokens: &[u32],
        labels: &[u32],
        batch: usize,
        length: usize,
        total: usize,
    ) -> Result<()> {
        self.forward_normalized(tokens, labels, batch, length, Some(total))?;
        self.add_backward()
    }

    /// One update of `fullGradientMomentumRun` from the accumulated data
    /// gradient: adds `penalty * x`, then the clipped momentum step.
    pub fn full_step(&mut self, penalty: f32, rate: f32, completed: usize) -> Result<()> {
        self.parameters.add_penalty(&self.kernel, penalty)?;
        self.parameters
            .update(&self.adam, &self.kernel, rate, completed)
    }

    /// Backpropagates the last forward pass and adds its parameter gradient.
    /// Every parameter adjoint accumulates: matrix products and gates add,
    /// and the LeRoPE, score and turn adjoints use atomic additions.
    fn add_backward(&mut self) -> Result<()> {
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
        self.tape.clear_gradients(&self.clear)?;
        // SAFETY: this launch uses the logit buffer extent.
        unsafe {
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
            let e = self.parameters.embedding();
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
            return Err("invalid Argand checkpoint".into());
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
    let mut initial = engine.weights()?;
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
    let extra_weights = match variant.routing {
        Routing::NormPhase => {
            norm_phase_checks::check(&engine)?;
            2 * engine.shape.layers
        }
        Routing::WriteGate => 4 * engine.shape.width * engine.shape.layers,
        Routing::ReadWriteGate | Routing::SkipWriteGate => {
            8 * engine.shape.width * engine.shape.layers
        }
        _ => 0,
    };
    if extra_weights != 0 {
        let mut ordinary = Engine::new(
            engine.shape.clone(),
            2,
            1,
            Variant {
                routing: Routing::Single,
                ..variant
            },
        )?;
        let shared = ordinary.weights()?;
        if initial.len() != shared.len() + extra_weights
            || initial[..shared.len()] != shared
            || initial[shared.len()..].iter().any(|&v| v != 0.0)
        {
            return Err("residual initialization differs from the matched ordinary model".into());
        }
        engine.forward(&tokens, &labels, 2, 3)?;
        ordinary.forward(&tokens, &labels, 2, 3)?;
        if engine.logits.read()? != ordinary.logits.read()? {
            return Err("zero residual parameters changed ordinary logits".into());
        }
        // Exercise the input and parameter adjoints away from the identity.
        for layer in &engine.parameters.layers {
            if let Some(at) = layer.turn {
                initial[at] = 0.13;
                initial[at + 1] = -0.17;
            }
            for gates in [
                layer.write_gates,
                layer.second_gate.map(SecondGate::offsets),
            ]
            .into_iter()
            .flatten()
            {
                for at in gates {
                    for (j, value) in initial[at..at + 2 * engine.shape.width]
                        .iter_mut()
                        .enumerate()
                    {
                        *value = 0.01 * ((j % 7) as f32 - 3.0);
                    }
                }
            }
        }
        engine.restore(&initial)?;
        println!("Zero residual parameters reproduce ordinary initialization and logits");
    }
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
    for at in engine
        .parameters
        .layers
        .iter()
        .filter_map(|layer| layer.turn)
    {
        if gradient[at..at + 2].iter().any(|g| g.abs() < 1e-6) {
            return Err("missing attention or FFN residual phase gradient".into());
        }
    }
    for gates in engine.parameters.layers.iter().flat_map(|layer| {
        [
            layer.write_gates,
            layer.second_gate.map(SecondGate::offsets),
        ]
        .into_iter()
        .flatten()
    }) {
        for at in gates {
            let g = &gradient[at..at + 2 * engine.shape.width];
            if (0..2).any(|coord| g[coord..].iter().step_by(2).all(|x| x.abs() < 1e-6)) {
                return Err("missing real or imaginary token-gate gradient".into());
            }
        }
    }
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
    let layer = &engine.parameters.layers[0];
    let trials =
        3 + usize::from(layer.write_gates.is_some()) + usize::from(layer.second_gate.is_some());
    for trial in 0..trials {
        let mut direction: Vec<_> = initial.iter().map(|_| rng.gauss(0.0, 1.0) as f32).collect();
        if trial >= 3 {
            direction[..initial.len() - extra_weights].fill(0.0);
            // Trial 3 probes write parameters alone. Trial 4 probes the second gate alone.
            for layer in &engine.parameters.layers {
                let excluded = if trial == 3 {
                    layer.second_gate.map(SecondGate::offsets)
                } else {
                    layer.write_gates
                };
                for at in excluded.into_iter().flatten() {
                    direction[at..at + 2 * engine.shape.width].fill(0.0);
                }
            }
        }
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
    // The two examples hold three and two of the five labels. Each part
    // divides by all five, so the parts sum to the batch gradient.
    engine.restore(&initial)?;
    engine.forward(&tokens, &labels, 2, 3)?;
    engine.backward()?;
    let whole = engine.parameters.grad.read()?;
    engine.zero_gradient()?;
    for rows in [0..3, 3..6] {
        engine.accumulate(&tokens[rows.clone()], &labels[rows], 1, 3, 5)?;
    }
    let parts = engine.parameters.grad.read()?;
    let largest = whole.iter().fold(0.0_f32, |m, g| m.max(g.abs()));
    let gap = parts
        .iter()
        .zip(&whole)
        .fold(0.0_f32, |m, (&a, &b)| m.max((a - b).abs()));
    if parts.len() != whole.len()
        || !parts.iter().all(|g| g.is_finite())
        || gap > 1e-5 * (1.0 + largest)
    {
        return Err("accumulated microbatch gradients disagree with the batch gradient".into());
    }
    println!(
        "Label-weighted microbatch gradients sum to the batch gradient within {gap:.1e} of {largest:.3}"
    );
    if matches!(variant.optimizer, Optimizer::SgdMomentum) {
        // The penalty enters the gradient once, before clipping. A zero rate
        // keeps the weights and sets the velocity to the clipped sum.
        let penalty = 0.5_f32;
        let before = engine.state()?;
        engine.full_step(penalty, 0.0, 0)?;
        let after = engine.state()?;
        let count = parts.len();
        let sum: Vec<f64> = parts
            .iter()
            .zip(&initial)
            .map(|(&g, &x)| f64::from(g) + f64::from(penalty) * f64::from(x))
            .collect();
        let norm = sum.iter().map(|g| g * g).sum::<f64>().sqrt();
        let scale = norm.max(1.0);
        if after[..count] != before[..count]
            || after[count..]
                .iter()
                .zip(&before[count..])
                .zip(&sum)
                .any(|((&v, &old), &g)| {
                    (f64::from(v) - (0.9 * f64::from(old) + g / scale)).abs() > 2e-6
                })
        {
            return Err("penalized full-batch step disagrees with the reference".into());
        }
        println!("Full-batch penalty enters once before clipping at gradient norm {norm:.4}");
    }
    match variant.optimizer {
        Optimizer::SgdMomentum => engine
            .parameters
            .check_sgd_momentum(&engine.adam, &engine.kernel)?,
        Optimizer::AdamW => {}
        Optimizer::AmsGradW(config) => {
            engine
                .parameters
                .check_amsgradw(&engine.adam, &engine.kernel, config)?
        }
    }
    println!(
        "{} vocabulary prediction, prefix causality and tied adjoints verified",
        variant.model()
    );
    Ok(())
}

//! A reverse pass over the kernels used by the Phasor v2 decoder.
use super::complex_mhc::{self, AMPLITUDES, Route, STREAMS};
use super::{F, P, Result, grid, kernels, scores};
use kernite_cuda::{Cuda, Grid};

#[derive(Clone, Copy)]
pub(super) struct Pair {
    pub x: P,
    pub dx: P,
    pub len: usize,
}
impl Pair {
    pub fn matrix(self, batches: usize, rows: usize, cols: usize) -> Matrix {
        assert_eq!(self.len, 2 * batches * rows * cols);
        Matrix {
            pair: self,
            batches,
            rows,
            cols,
            transposed: false,
            conjugated: false,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Matrix {
    pair: Pair,
    batches: usize,
    rows: usize,
    cols: usize,
    transposed: bool,
    conjugated: bool,
}
impl Matrix {
    pub fn transpose(mut self) -> Self {
        self.transposed = !self.transposed;
        std::mem::swap(&mut self.rows, &mut self.cols);
        self
    }

    pub fn conjugate(mut self) -> Self {
        self.conjugated = !self.conjugated;
        self
    }

    fn adjoint(self) -> Self {
        self.transpose().conjugate()
    }
}

#[derive(Clone, Copy)]
enum Operation {
    Matrix(Matrix, Matrix, Matrix),
    Rms(Pair, Pair, usize),
    TurnByNorm(Pair, Pair, Pair, usize),
    Product(Pair, Pair, Pair),
    Sum(Pair, Pair, Pair, f32),
    Heads(Pair, Pair, bool),
    Rotate(Pair, Pair),
    Scores(Pair, Pair, Pair),
    Expand(Pair, Pair, usize),
    Collapse(Pair, Pair, usize),
    Route(Route),
}

struct Slot {
    x: F,
    dx: F,
    len: usize,
}

#[derive(Default)]
pub(super) struct Tape {
    slots: Vec<Slot>,
    used: usize,
    operations: Vec<Operation>,
}

pub(super) struct Context<'a> {
    pub cuda: &'a Cuda,
    pub kernel: &'a kernels::device::Kernels,
    pub scores: &'a scores::device::Kernels,
    pub mhc: Option<&'a complex_mhc::Kernels>,
    pub rotations: P,
    pub alpha: Pair,
    pub length: usize,
    pub heads: usize,
    pub head_width: usize,
}

impl Tape {
    pub fn reset(&mut self) {
        self.used = 0;
        self.operations.clear();
    }

    pub fn allocate(&mut self, cuda: &Cuda, len: usize) -> Result<Pair> {
        if self.slots.get(self.used).is_none() {
            self.slots.push(Slot {
                x: cuda.alloc(len)?,
                dx: cuda.alloc(len)?,
                len,
            });
        }
        let slot = &mut self.slots[self.used];
        if slot.x.len() < len {
            *slot = Slot {
                x: cuda.alloc(len)?,
                dx: cuda.alloc(len)?,
                len,
            };
        }
        slot.len = len;
        self.used += 1;
        Ok(Pair {
            x: slot.x.ptr(),
            dx: slot.dx.ptr(),
            len,
        })
    }

    pub fn clear_gradients(&self, clear: &crate::slot_copy::kernels::Kernels) -> Result<()> {
        for slot in &self.slots[..self.used] {
            // SAFETY: the grid is bounded by the used portion of this slot.
            unsafe {
                clear.clear(grid(slot.len), slot.dx.ptr(), slot.len as u32)?;
            }
        }
        Ok(())
    }

    fn apply(&mut self, ctx: &Context<'_>, op: Operation, output: Pair) -> Result<Pair> {
        ctx.apply::<false>(op)?;
        self.operations.push(op);
        Ok(output)
    }

    pub fn mm(&mut self, ctx: &Context<'_>, a: Matrix, b: Matrix) -> Result<Pair> {
        assert_eq!((a.batches, a.cols), (b.batches, b.rows));
        let out = self.allocate(ctx.cuda, 2 * a.batches * a.rows * b.cols)?;
        self.apply(
            ctx,
            Operation::Matrix(out.matrix(a.batches, a.rows, b.cols), a, b),
            out,
        )
    }

    pub fn rms(&mut self, ctx: &Context<'_>, x: Pair, width: usize) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len)?;
        self.apply(ctx, Operation::Rms(out, x, width), out)
    }

    pub fn turn_by_norm(
        &mut self,
        ctx: &Context<'_>,
        x: Pair,
        theta: Pair,
        width: usize,
    ) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len)?;
        self.apply(ctx, Operation::TurnByNorm(out, x, theta, width), out)
    }

    pub fn product(&mut self, ctx: &Context<'_>, a: Pair, b: Pair) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, a.len)?;
        self.apply(ctx, Operation::Product(out, a, b), out)
    }

    /// Multiply each normalized branch write by 1 + sum_j input[j] * weights[j].
    /// The complex projection produces one scalar per token, shared by its channels.
    pub fn gate_write(
        &mut self,
        ctx: &Context<'_>,
        branch: Pair,
        input: Pair,
        weights: Option<Pair>,
    ) -> Result<Pair> {
        let Some(weights) = weights else {
            return Ok(branch);
        };
        let width = weights.len / 2;
        let rows = input.len / weights.len;
        let gate = self.mm(
            ctx,
            input.matrix(1, rows, width),
            weights.matrix(1, width, 1),
        )?;
        let correction = self.mm(ctx, gate.matrix(rows, 1, 1), branch.matrix(rows, 1, width))?;
        self.sum(ctx, branch, correction, 1.0)
    }

    pub fn sum(&mut self, ctx: &Context<'_>, a: Pair, b: Pair, scale: f32) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, a.len)?;
        self.apply(ctx, Operation::Sum(out, a, b, scale), out)
    }

    pub fn heads(&mut self, ctx: &Context<'_>, x: Pair, flip: bool) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len)?;
        self.apply(ctx, Operation::Heads(out, x, flip), out)
    }

    pub fn rotate(&mut self, ctx: &Context<'_>, x: Pair) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len)?;
        self.apply(ctx, Operation::Rotate(out, x), out)
    }

    pub fn scores(&mut self, ctx: &Context<'_>, x: Pair, s: Pair) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len)?;
        self.apply(ctx, Operation::Scores(out, x, s), out)
    }

    pub fn expand(&mut self, ctx: &Context<'_>, x: Pair, width: usize) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, STREAMS * x.len)?;
        self.apply(ctx, Operation::Expand(out, x, width), out)
    }

    pub fn collapse(&mut self, ctx: &Context<'_>, x: Pair, width: usize) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, x.len / STREAMS)?;
        self.apply(ctx, Operation::Collapse(out, x, width), out)
    }

    pub fn route(&mut self, ctx: &Context<'_>, logits: Pair, rows: usize) -> Result<Route> {
        let route = Route {
            logits,
            packed: self.allocate(ctx.cuda, rows * AMPLITUDES)?,
            amplitudes: self.allocate(ctx.cuda, rows * AMPLITUDES)?,
            pre: self.allocate(ctx.cuda, 2 * rows * STREAMS)?,
            post: self.allocate(ctx.cuda, 2 * rows * STREAMS)?,
            residual: self.allocate(ctx.cuda, 2 * rows * STREAMS * STREAMS)?,
            edge_gradient: (logits.len / rows == AMPLITUDES + 3 * STREAMS)
                .then(|| self.allocate(ctx.cuda, rows * STREAMS * STREAMS))
                .transpose()?,
        };
        let op = Operation::Route(route);
        ctx.apply::<false>(op)?;
        self.operations.push(op);
        Ok(route)
    }

    pub fn backward(&self, ctx: &Context<'_>) -> Result<()> {
        for &op in self.operations.iter().rev() {
            ctx.apply::<true>(op)?;
        }
        Ok(())
    }
}

impl Context<'_> {
    fn matmul(&self, y: Matrix, a: Matrix, b: Matrix, pointers: [P; 3], add: bool) -> Result<()> {
        let tiles_m = a.rows.div_ceil(16);
        let tiles_n = b.cols.div_ceil(16);
        let flags = u32::from(a.transposed)
            | u32::from(b.transposed) << 1
            | u32::from(y.transposed) << 2
            | u32::from(a.conjugated) << 3
            | u32::from(b.conjugated) << 4
            | u32::from(y.conjugated) << 5
            | u32::from(add) << 6;
        // SAFETY: all views describe complete batches of equally sized contiguous matrices.
        unsafe {
            self.kernel.matmul(
                Grid::new((tiles_m * tiles_n * a.batches) as u32, 256),
                pointers[0],
                pointers[1],
                pointers[2],
                a.rows as u32,
                b.cols as u32,
                a.cols as u32,
                tiles_m as u32,
                tiles_n as u32,
                flags,
            )?;
        }
        Ok(())
    }

    fn apply<const BACK: bool>(&self, op: Operation) -> Result<()> {
        let k = self.kernel;
        let (d, t, h) = (
            self.head_width as u32,
            self.length as u32,
            self.heads as u32,
        );
        // SAFETY: tape nodes own equal-sized coordinate buffers. Each launch
        // covers just the explicitly supplied tensor dimensions, in one stream.
        unsafe {
            match op {
                Operation::Matrix(y, a, b) if BACK => {
                    self.matmul(a, y, b.adjoint(), [a.pair.dx, y.pair.dx, b.pair.x], true)?;
                    self.matmul(b, a.adjoint(), y, [b.pair.dx, a.pair.x, y.pair.dx], true)?;
                }
                Operation::Matrix(y, a, b) => {
                    self.matmul(y, a, b, [y.pair.x, a.pair.x, b.pair.x], false)?;
                }
                Operation::Rms(y, x, width) => {
                    let launch = Grid::new((x.len / (2 * width)) as u32, 32);
                    if BACK {
                        k.rms::<true>(
                            launch,
                            x.dx,
                            x.x,
                            y.dx,
                            (2 * width) as u32,
                            (width as f32).sqrt(),
                        )?;
                    } else {
                        k.rms::<false>(
                            launch,
                            y.x,
                            x.x,
                            y.dx,
                            (2 * width) as u32,
                            (width as f32).sqrt(),
                        )?;
                    }
                }
                Operation::TurnByNorm(y, x, theta, width) => {
                    let launch = Grid::new((x.len / (2 * width)) as u32, 32);
                    k.turn_by_norm::<BACK>(
                        launch,
                        y.x,
                        x.x,
                        y.dx,
                        x.dx,
                        theta.x,
                        theta.dx,
                        width as u32,
                    )?;
                }
                Operation::Product(y, a, b) => {
                    let launch = grid(y.len / 2);
                    let size = (y.len / 2) as u32;
                    if BACK {
                        k.product::<true>(launch, y.x, a.x, b.x, y.dx, a.dx, b.dx, size)?;
                    } else {
                        k.product::<false>(launch, y.x, a.x, b.x, y.dx, a.dx, b.dx, size)?;
                    }
                }
                Operation::Sum(y, a, b, scale) => {
                    let launch = grid(y.len);
                    if BACK {
                        k.sum::<true>(
                            launch,
                            y.x,
                            a.x,
                            b.x,
                            y.dx,
                            a.dx,
                            b.dx,
                            y.len as u32,
                            scale,
                        )?;
                    } else {
                        k.sum::<false>(
                            launch,
                            y.x,
                            a.x,
                            b.x,
                            y.dx,
                            a.dx,
                            b.dx,
                            y.len as u32,
                            scale,
                        )?;
                    }
                }
                Operation::Heads(y, x, flip) => {
                    let launch = grid(y.len);
                    let size = y.len as u32;
                    match (BACK, flip) {
                        (false, false) => {
                            k.heads::<false, false>(launch, y.x, x.x, size, t, h, 2 * d)?
                        }
                        (false, true) => {
                            k.heads::<false, true>(launch, y.x, x.x, size, t, h, 2 * d)?
                        }
                        (true, false) => {
                            k.heads::<true, false>(launch, x.dx, y.dx, size, t, h, 2 * d)?
                        }
                        (true, true) => {
                            k.heads::<true, true>(launch, x.dx, y.dx, size, t, h, 2 * d)?
                        }
                    }
                }
                Operation::Rotate(y, x) => {
                    let size = (y.len / 2) as u32;
                    let launch = grid(size as usize);
                    if BACK {
                        k.rotate::<true>(
                            launch,
                            x.dx,
                            y.x,
                            y.dx,
                            self.rotations,
                            self.alpha.dx,
                            d,
                            t,
                            h,
                            size,
                        )?;
                    } else {
                        k.rotate::<false>(
                            launch,
                            y.x,
                            x.x,
                            y.dx,
                            self.rotations,
                            self.alpha.dx,
                            d,
                            t,
                            h,
                            size,
                        )?;
                    }
                }
                Operation::Scores(y, x, s) => {
                    let launch = Grid::new((y.len / (2 * t as usize)) as u32, 32);
                    let gain = std::f32::consts::LN_2 / (d as f32).sqrt();
                    if BACK {
                        self.scores
                            .weights::<true>(launch, y.x, s.x, x.x, y.dx, x.dx, s.dx, t, gain)?;
                    } else {
                        self.scores
                            .weights::<false>(launch, y.x, s.x, x.x, y.dx, x.dx, s.dx, t, gain)?;
                    }
                }
                Operation::Expand(y, x, width) => {
                    let mhc = &self.mhc.expect("stream expansion requires mHC").real;
                    let launch = grid(x.len);
                    let (width, count) = ((2 * width) as u32, x.len as u32);
                    // The embedding RMS has no other consumers in an mHC run.
                    if BACK {
                        mhc.expand::<true>(launch, x.dx, y.dx, width, count)?;
                    } else {
                        mhc.expand::<false>(launch, y.x, x.x, width, count)?;
                    }
                }
                Operation::Collapse(y, x, width) => {
                    let mhc = &self.mhc.expect("stream collapse requires mHC").real;
                    let launch = grid(y.len);
                    let (width, count) = ((2 * width) as u32, y.len as u32);
                    // The final four streams have no other consumers.
                    if BACK {
                        mhc.collapse::<true>(launch, x.dx, y.dx, width, count)?;
                    } else {
                        mhc.collapse::<false>(launch, y.x, x.x, width, count)?;
                    }
                }
                Operation::Route(route) => {
                    route.apply::<BACK>(self.mhc.expect("missing mHC kernels"))?
                }
            }
        }
        Ok(())
    }
}

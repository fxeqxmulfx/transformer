//! A reverse pass over the kernels used by the Argand decoder.
use super::{F, P, Result, gemm, grid, kernels, scores};
use crate::random::Random;
use gemm::device as g;
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
    Product(Pair, Pair, Pair),
    Sum(Pair, Pair, Pair, f32),
    Heads(Pair, Pair, bool),
    Rotate(Pair, Pair),
    Scores(Pair, Pair, Pair),
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
    pub gemm: &'a gemm::device::Kernels,
    /// Room for the partial products of a split `k`, in floats.
    pub workspace: P,
    pub workspace_len: usize,
    pub multiprocessors: usize,
    pub scores: &'a scores::device::Kernels,
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

    pub fn clear_gradients(&self, kernel: &kernels::device::Kernels) -> Result<()> {
        for slot in &self.slots[..self.used] {
            // SAFETY: the grid is bounded by the used portion of this slot.
            unsafe {
                kernel.clear(grid(slot.len), slot.dx.ptr(), slot.len as u32)?;
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

    pub fn product(&mut self, ctx: &Context<'_>, a: Pair, b: Pair) -> Result<Pair> {
        let out = self.allocate(ctx.cuda, a.len)?;
        self.apply(ctx, Operation::Product(out, a, b), out)
    }

    /// Multiply each row by 1 + sum_j input[j] * weights[j].
    /// The complex projection produces one scalar per token, shared by its channels.
    pub fn gate_rows(
        &mut self,
        ctx: &Context<'_>,
        values: Pair,
        input: Pair,
        weights: Pair,
    ) -> Result<Pair> {
        let width = weights.len / 2;
        let rows = input.len / weights.len;
        let gate = self.mm(
            ctx,
            input.matrix(1, rows, width),
            weights.matrix(1, width, 1),
        )?;
        let correction = self.mm(ctx, gate.matrix(rows, 1, 1), values.matrix(rows, 1, width))?;
        self.sum(ctx, values, correction, 1.0)
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

    /// The real operations of the complex matrix products on the tape: eight
    /// for each complex multiply-add, in the product and in its two adjoints.
    pub fn product_flops(&self) -> f64 {
        self.operations
            .iter()
            .map(|op| match op {
                Operation::Matrix(_, a, b) => 24.0 * (a.batches * a.rows * a.cols * b.cols) as f64,
                _ => 0.0,
            })
            .sum()
    }

    pub fn backward(&self, ctx: &Context<'_>) -> Result<()> {
        for &op in self.operations.iter().rev() {
            ctx.apply::<true>(op)?;
        }
        Ok(())
    }
}

impl Context<'_> {
    /// `y (+)= a b` for views of complex matrices. Outputs of at least 32
    /// rows and columns take the tiled product; when its tiles alone would
    /// leave the GPU idle, `k` is split into parts of at least 256 steps,
    /// whose partial products the workspace holds and `reduce` adds in order.
    fn matmul(&self, y: Matrix, a: Matrix, b: Matrix, pointers: [P; 3], add: bool) -> Result<()> {
        let (m, n, k, batches) = (a.rows, b.cols, a.cols, a.batches);
        if m < 32 || n < 32 {
            return self.small_matmul(y, a, b, pointers, add);
        }
        let tiles_n = n.div_ceil(g::TILE);
        let blocks = m.div_ceil(g::TILE) * tiles_n * batches;
        let target = 2 * self.multiprocessors;
        let parts = if blocks >= target {
            1
        } else {
            target.div_ceil(blocks).min(k / 256).max(1)
        };
        let span = k.div_ceil(parts).next_multiple_of(g::DEPTH).max(g::DEPTH);
        let parts = k.div_ceil(span).max(1);
        let output = u32::from(y.conjugated) * g::CONJUGATE_Y
            | u32::from(y.transposed) * g::TRANSPOSE_Y
            | u32::from(add) * g::ADD;
        let mut flags =
            u32::from(a.conjugated) * g::CONJUGATE_A | u32::from(b.conjugated) * g::CONJUGATE_B;
        let destination = if parts > 1 {
            flags |= g::PARTIAL;
            assert!(2 * parts * batches * m * n <= self.workspace_len);
            self.workspace
        } else {
            flags |= output;
            pointers[0]
        };
        let launch = Grid {
            blocks: [(blocks / batches) as u32, parts as u32, batches as u32],
            threads: [256, 1, 1],
            shared: 0,
        };
        let args = (
            destination,
            pointers[1],
            pointers[2],
            m as u32,
            n as u32,
            k as u32,
            tiles_n as u32,
            span as u32,
            flags,
        );
        // SAFETY: the views describe complete batches of contiguous
        // matrices, and the partial products fit in the workspace.
        unsafe {
            let (d, x, w, rows, cols, depth, t, s, f) = args;
            match (a.transposed, b.transposed) {
                (false, false) => self.gemm.cgemm::<false, false>(launch, d, x, w, rows, cols, depth, t, s, f)?,
                (true, false) => self.gemm.cgemm::<true, false>(launch, d, x, w, rows, cols, depth, t, s, f)?,
                (false, true) => self.gemm.cgemm::<false, true>(launch, d, x, w, rows, cols, depth, t, s, f)?,
                (true, true) => self.gemm.cgemm::<true, true>(launch, d, x, w, rows, cols, depth, t, s, f)?,
            }
            if parts > 1 {
                let count = batches * m * n;
                self.gemm.reduce(
                    grid(count),
                    pointers[0],
                    self.workspace,
                    m as u32,
                    n as u32,
                    count as u32,
                    parts as u32,
                    output,
                )?;
            }
        }
        Ok(())
    }

    fn small_matmul(&self, y: Matrix, a: Matrix, b: Matrix, pointers: [P; 3], add: bool) -> Result<()> {
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
            }
        }
        Ok(())
    }
}

impl Context<'_> {
    /// Compares the tiled product with `small_matmul` on every view of the
    /// operands: transposed or not, conjugated or not, added or stored. With
    /// `k` whole the two agree exactly, as the chains are the same; with `k`
    /// split they agree to rounding. Returns the largest split difference.
    pub fn check_products(&self) -> Result<f32> {
        let mut rng = Random::new(7);
        let mut largest = 0.0_f32;
        // (m, n, k, batches): edges, several batches, and two splits of k.
        for (m, n, k, batches) in [
            (64, 64, 8, 1),
            (100, 37, 13, 2),
            (33, 130, 300, 1),
            (40, 96, 7, 3),
            (128, 128, 1024, 1),
            (64, 64, 2000, 1),
        ] {
            let mut data =
                |len: usize| -> Vec<f32> { (0..2 * len).map(|_| rng.gauss(0.0, 1.0) as f32).collect() };
            let (a, b, start) = (
                self.cuda.upload(&data(batches * m * k))?,
                self.cuda.upload(&data(batches * k * n))?,
                data(batches * m * n),
            );
            let split = m.div_ceil(g::TILE) * n.div_ceil(g::TILE) * batches < 2 * self.multiprocessors
                && k >= 512;
            for flags in 0..128_u32 {
                let bit = |b: u32| flags & (1 << b) != 0;
                let view = |buffer: &F, rows, cols, t: bool, c: bool| {
                    let pair = Pair {
                        x: buffer.ptr(),
                        dx: buffer.ptr(),
                        len: buffer.len(),
                    };
                    let v = if t {
                        pair.matrix(batches, cols, rows).transpose()
                    } else {
                        pair.matrix(batches, rows, cols)
                    };
                    if c { v.conjugate() } else { v }
                };
                let (small, tiled) = (self.cuda.upload(&start)?, self.cuda.upload(&start)?);
                for (out, tiled_kernel) in [(&small, false), (&tiled, true)] {
                    let (y, x, w) = (
                        view(out, m, n, bit(2), bit(5)),
                        view(&a, m, k, bit(0), bit(3)),
                        view(&b, k, n, bit(1), bit(4)),
                    );
                    let pointers = [out.ptr(), a.ptr(), b.ptr()];
                    if tiled_kernel {
                        self.matmul(y, x, w, pointers, bit(6))?;
                    } else {
                        self.small_matmul(y, x, w, pointers, bit(6))?;
                    }
                }
                let (small, tiled) = (small.read()?, tiled.read()?);
                let scale = small.iter().fold(1.0_f32, |s, v| s.max(v.abs()));
                let gap = small
                    .iter()
                    .zip(&tiled)
                    .fold(0.0_f32, |g, (x, y)| if x == y { g } else { g.max((x - y).abs()).max(f32::MIN_POSITIVE) });
                if (!split && gap != 0.0) || gap > 1e-5 * scale || tiled.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "tiled product disagrees for m={m} n={n} k={k} batches={batches} flags={flags:#09b}: {gap:e}"
                    )
                    .into());
                }
                largest = largest.max(gap / scale);
            }
        }
        Ok(largest)
    }
}

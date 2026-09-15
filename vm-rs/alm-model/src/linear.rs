//! The linear algebra of the forward pass, which is where its time goes.
//!
//! Split out of `model.rs` because `perf` says so: on the `sudoku` trace the
//! inner product of `Dense::apply` carries 48 % of the whole run's samples and
//! `SparseHead::argmax` another 6 %, against 21 % for the hull.  A kernel that
//! large deserves to be a named unit with its own benchmark
//! (`alm-model/benches/linear.rs`) rather than a private helper.
//!
//! The summation order is part of the answer and not of the schedule.  Float
//! addition is not associative, so `transformer.cpp` summing a row left to
//! right is a specification: any regrouping of one row's terms would produce a
//! different `f64` and a different reference trace.  What may move is the
//! order *between* rows, which share nothing.
//!
//! Both kernels are formalized: `Transformer.ALM.Dense` for the interleaved
//! layout and `Transformer.ALM.SparseHead` for the compressed head.  Each is
//! stated over an *arbitrary* `add` and `mul`, with no associativity and no
//! commutativity assumed, because that is precisely the strength of the claim
//! being made — the answer is unchanged step for step, in `f64` as in `R`,
//! rather than to within a rounding.

/// How many rows are summed at once: the width of one AVX2 register in `f64`.
///
/// Four independent accumulators is also roughly what it takes to cover the
/// four-cycle latency of an FMA, so the number would be about this even on a
/// machine with no vector unit at all.
const LANES: usize = 8;

/// A dense matrix, `[rows, cols]` as `model.bin` stores it, held four rows at
/// a time.
///
/// The file is row-major and the obvious loop follows it: one row, one
/// accumulator, `cols` dependent additions into it.  At `cols = 38` that is
/// thirty-eight FMAs in a chain four cycles deep and one core that can retire
/// two of them a cycle — a ninety per cent idle pipeline, and `perf` duly
/// finds forty-eight per cent of a run inside it.
///
/// So the weights are interleaved instead: `w[(b * cols + j) * LANES + l]`
/// holds row `b * LANES + l`, column `j`, which puts the `j`-th entry of four
/// consecutive rows in four adjacent words.  One load, one broadcast of `x[j]`
/// and one vector FMA then advance four rows at once, each lane carrying its
/// own accumulator.
///
/// This changes no answer, and the reason is the layout rather than an
/// analysis: each lane still sums its own row left to right, in the order
/// `transformer.cpp` sums it, so every partial sum is the `f64` it was before
/// and so is the total.  It is the order *between* rows that moves, and rows
/// share nothing.  The reference traces are bit-for-bit what they were.
///
/// `Transformer.ALM.unpack_packed` is the half a mistyped index would break:
/// the scatter is injective, proved by exhibiting the inverse that reads a
/// row and a column back out of an address, and `packed_lt_buffer` puts every
/// address it writes inside the buffer allocated below.
///
/// Rows are padded up to a multiple of `LANES` with zeros; their sums are
/// computed and thrown away, which costs at most three rows of a matrix and
/// removes the remainder loop from the hot path.
pub struct Dense {
    rows: usize,
    cols: usize,
    w: Vec<f64>,
}

impl Dense {
    pub fn of(w: &[f64], rows: usize, cols: usize) -> Dense {
        assert_eq!(w.len(), rows * cols, "the weight file declares its own shapes");
        let mut packed = vec![0.0; rows.div_ceil(LANES) * cols * LANES];
        for (i, row) in w.chunks_exact(cols).enumerate() {
            let (b, l) = (i / LANES, i % LANES);
            for (j, &a) in row.iter().enumerate() {
                packed[(b * cols + j) * LANES + l] = a;
            }
        }
        Dense { rows, cols, w: packed }
    }

    /// `y = W x`, each row summed left to right, four rows at a time.
    ///
    /// The order within a row is the one `transformer.cpp` uses and the one
    /// the reference traces were generated under; float addition is not
    /// associative, so it is part of the answer rather than of the schedule.
    ///
    /// `Transformer.ALM.apply_eq_rowMajor` is that sentence as a theorem: the
    /// packed row and the file's row fold to the same element for *every*
    /// `add` and `mul` whatsoever.  A kernel that regrouped a row could not
    /// satisfy a statement that weak in its arithmetic.
    ///
    /// The padding is `Transformer.ALM.DensePad`, and it is a fact about
    /// lengths rather than about sums -- over an arbitrary `add` and `mul` the
    /// padded lanes do not come to zero, and they do not have to, because they
    /// are dropped.  Block and lane are the quotient and the remainder of the
    /// row index, so every row is written by exactly one lane of exactly one
    /// block (`chunkOf_writeAt`, `chunkOf_mem`); a lane this line copies is a
    /// row the matrix has (`writeAt_lt_rows`) and a lane it truncates is past
    /// the last one (`dropped_ge_rows`).  `blocks_of_buffer` is that the two
    /// sides of the `zip` have the same number of steps, so neither loop ends
    /// early, and `rows_le_blockCount_mul` is the hypothesis `packed_lt_buffer`
    /// took on faith, discharged for the allocation `of` really makes.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols);
        debug_assert_eq!(y.len(), self.rows);
        let (blocks, _) = self.w.as_chunks::<LANES>();
        for (blk, out) in blocks.chunks_exact(self.cols).zip(y.chunks_mut(LANES)) {
            let mut s = [0.0; LANES];
            for (col, &b) in blk.iter().zip(x) {
                for l in 0..LANES {
                    s[l] += col[l] * b;
                }
            }
            out.copy_from_slice(&s[..out.len()]);
        }
    }
}

/// The output head, in compressed sparse rows.
///
/// It is the one projection the C++ runtime does not do densely, and the
/// reason is in the numbers: the head is `vocab x d_model`, 915 x 38 here and
/// 85 % zero, and it runs once per generated token.  Skipping the zeros is
/// exact — adding `0.0 * x` to a partial sum never changes it — so this is
/// the same argmax, not an approximation of it.
///
/// `Transformer.ALM.sparseFold_eq_rowFold` is that claim, and carrying it
/// through Lean sharpened the condition it rests on: `0.0 * x` is `NaN` when
/// `x` is infinite or `NaN`, which destroys the partial sum rather than
/// preserving it.  So the hypothesis is quantified over the *entries of the
/// input*, and it is the activations reaching the head that have to be
/// finite — not, as an earlier wording here had it, the partial sum.
pub struct SparseHead {
    rows: usize,
    /// `row i` occupies `col[ptr[i]..ptr[i+1]]`.
    ptr: Vec<usize>,
    col: Vec<usize>,
    val: Vec<f64>,
}

impl SparseHead {
    pub fn of(w: &[f64], rows: usize, cols: usize) -> SparseHead {
        let mut head = SparseHead { rows, ptr: vec![0], col: Vec::new(), val: Vec::new() };
        for i in 0..rows {
            for j in 0..cols {
                let v = w[i * cols + j];
                if v != 0.0 {
                    head.col.push(j);
                    head.val.push(v);
                }
            }
            head.ptr.push(head.col.len());
        }
        head
    }

    /// The first index attaining the maximum, as `Tensor::argmax` and the C++
    /// loop both resolve it.
    ///
    /// The strict `>` is what resolves it that way, and the reference traces
    /// depend on the choice.  `Transformer.ALM.firstMax` is this loop,
    /// `firstMax_le` that its answer is a maximum and `firstMax_first` that
    /// nothing before it attains one; `firstMax_sparse_eq` composes them with
    /// the line above, so the compressed head returns the dense head's index
    /// and not merely an equally good one.
    pub fn argmax(&self, x: &[f64]) -> usize {
        let mut best = 0;
        let mut best_score = f64::NEG_INFINITY;
        for i in 0..self.rows {
            let mut s = 0.0;
            for k in self.ptr[i]..self.ptr[i + 1] {
                s += self.val[k] * x[self.col[k]];
            }
            if s > best_score {
                best_score = s;
                best = i;
            }
        }
        best
    }

    /// The rows of the head, which is the vocabulary.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// How many of its entries are nonzero.
    pub fn nnz(&self) -> usize {
        self.val.len()
    }
}

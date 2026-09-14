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

/// A dense matrix, row-major `[rows, cols]`, exactly as `model.bin` stores it.
pub struct Dense {
    cols: usize,
    w: Vec<f64>,
}

impl Dense {
    pub fn of(w: &[f64], rows: usize, cols: usize) -> Dense {
        assert_eq!(w.len(), rows * cols, "the weight file declares its own shapes");
        Dense { cols, w: w.to_vec() }
    }

    /// `y = W x`, each row summed left to right.
    ///
    /// The order is the one `transformer.cpp` uses and the one the reference
    /// traces were generated under; float addition is not associative, so it
    /// is part of the answer rather than of the schedule.
    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        debug_assert_eq!(x.len(), self.cols);
        for (row, out) in self.w.chunks_exact(self.cols).zip(y.iter_mut()) {
            let mut s = 0.0;
            for (a, b) in row.iter().zip(x) {
                s += a * b;
            }
            *out = s;
        }
    }
}

/// The output head, in compressed sparse rows.
///
/// It is the one projection the C++ runtime does not do densely, and the
/// reason is in the numbers: the head is `vocab x d_model`, 915 x 38 here and
/// 85 % zero, and it runs once per generated token.  Skipping the zeros is
/// exact — adding `0.0 * x` to a finite partial sum never changes it — so
/// this is the same argmax, not an approximation of it.
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

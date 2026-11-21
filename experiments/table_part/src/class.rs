//! The class `g(u) + h(v) + uᵀ M v` over `F_p` on the table `F_p^k × F_p^l` (NV1), and two
//! independent counts of what a part of the table fixes: the rank of the evaluation matrix by
//! elimination, and `n − c + s` from the graph of the part.

const NO_PIVOT: usize = usize::MAX;

/// A table of the class, for `p < 128`. Rows and columns are numbered in base `p`, first
/// coordinate first, and the cell `(u, v)` has the index `u · p^l + v`.
pub struct Table {
    pub p: u8,
    pub k: usize,
    pub l: usize,
    pub rows: usize,
    pub columns: usize,
    /// `vec(u vᵀ)` of every cell, `k · l` bytes each.
    products: Vec<u8>,
}

impl Table {
    /// O(p^(k+l) · k · l) time and memory.
    pub fn new(p: u8, k: usize, l: usize) -> Self {
        let base = usize::from(p);
        let (rows, columns) = (base.pow(k as u32), base.pow(l as u32));
        let digit =
            |x: usize, len: usize, i: usize| (x / base.pow((len - 1 - i) as u32) % base) as u16;
        let mut products = Vec::with_capacity(rows * columns * k * l);
        for u in 0..rows {
            for v in 0..columns {
                for i in 0..k {
                    for j in 0..l {
                        products.push((digit(u, k, i) * digit(v, l, j) % u16::from(p)) as u8);
                    }
                }
            }
        }
        Self { p, k, l, rows, columns, products }
    }

    pub fn cells(&self) -> usize {
        self.rows * self.columns
    }

    /// The `n = p^k + p^l` rows and columns.
    pub fn lines(&self) -> usize {
        self.rows + self.columns
    }

    /// The number of entries of `M`.
    pub fn couplings(&self) -> usize {
        self.k * self.l
    }

    /// `dim H = n − 1 + k l`.
    pub fn dim(&self) -> usize {
        self.lines() - 1 + self.couplings()
    }

    pub fn product(&self, cell: usize) -> &[u8] {
        let size = self.couplings();
        &self.products[cell * size..(cell + 1) * size]
    }

    /// The cross `{(u, 0)} ∪ {(0, v)}` with the cells `(e_i, e_j)`: `dim H` cells.
    pub fn least_part(&self) -> Vec<u32> {
        let base = usize::from(self.p);
        let mut part: Vec<u32> = (0..self.rows).map(|u| (u * self.columns) as u32).collect();
        part.extend(1..self.columns as u32);
        for i in 0..self.k {
            for j in 0..self.l {
                let u = base.pow((self.k - 1 - i) as u32);
                let v = base.pow((self.l - 1 - j) as u32);
                part.push((u * self.columns + v) as u32);
            }
        }
        part
    }
}

/// A row-reduced span over `F_p`. `insert` is O(rank · width).
pub struct Span {
    p: u16,
    width: usize,
    row_of_pivot: Vec<usize>,
    basis: Vec<u8>,
    inverse: Vec<u8>,
    pub rank: usize,
}

impl Span {
    pub fn new(p: u8, width: usize) -> Self {
        let modulus = u16::from(p);
        let inverse = (0..modulus)
            .map(|x| (1..modulus).find(|y| x * y % modulus == 1).unwrap_or(0) as u8)
            .collect();
        Self { p: modulus, width, row_of_pivot: vec![NO_PIVOT; width], basis: vec![], inverse, rank: 0 }
    }

    pub fn clear(&mut self) {
        self.row_of_pivot.fill(NO_PIVOT);
        self.basis.clear();
        self.rank = 0;
    }

    /// Adds `row`, which it overwrites, and says whether the rank grew.
    pub fn insert(&mut self, row: &mut [u8]) -> bool {
        let p = self.p;
        for c in 0..self.width {
            let x = row[c];
            if x == 0 {
                continue;
            }
            let at = self.row_of_pivot[c];
            if at == NO_PIVOT {
                let scale = u16::from(self.inverse[usize::from(x)]);
                for y in &mut row[c..] {
                    *y = (u16::from(*y) * scale % p) as u8;
                }
                self.row_of_pivot[c] = self.rank;
                self.basis.extend_from_slice(row);
                self.rank += 1;
                return true;
            }
            let factor = p - u16::from(x);
            let pivot = &self.basis[at * self.width + c..(at + 1) * self.width];
            for (y, &z) in row[c..].iter_mut().zip(pivot) {
                *y = ((u16::from(*y) + factor * u16::from(z)) % p) as u8;
            }
        }
        false
    }
}

/// Buffers for the two counts, reused from part to part.
pub struct Workspace {
    parent: Vec<u32>,
    weight: Vec<u32>,
    offset: Vec<u8>,
    left: Vec<u8>,
    right: Vec<u8>,
    twist: Vec<u8>,
    twists: Span,
    design: Span,
    row: Vec<u8>,
}

impl Workspace {
    pub fn new(t: &Table) -> Self {
        let (n, size) = (t.lines(), t.couplings());
        Self {
            parent: vec![0; n],
            weight: vec![0; n],
            offset: vec![0; n * size],
            left: vec![0; size],
            right: vec![0; size],
            twist: vec![0; size],
            twists: Span::new(t.p, size),
            design: Span::new(t.p, n + size),
            row: vec![0; n + size],
        }
    }
}

/// The rank of the evaluation matrix `[1_u | 1_v | vec(u vᵀ)]` on `part`, by elimination. It
/// knows nothing of graphs. O(|part| · rank · (n + k l)) time.
pub fn design_rank(t: &Table, part: &[u32], w: &mut Workspace) -> usize {
    w.design.clear();
    let n = t.lines();
    for &cell in part {
        let cell = cell as usize;
        w.row.fill(0);
        w.row[cell / t.columns] = 1;
        w.row[t.rows + cell % t.columns] = 1;
        w.row[n..].copy_from_slice(t.product(cell));
        w.design.insert(&mut w.row);
        if w.design.rank == t.dim() {
            break;
        }
    }
    w.design.rank
}

/// The root of `x`, with `value(x) − value(root)` written to `out`. O(log n · size).
fn locate(parent: &[u32], offset: &[u8], p: u8, mut x: usize, out: &mut [u8]) -> usize {
    let size = out.len();
    out.fill(0);
    while parent[x] as usize != x {
        for (o, &d) in out.iter_mut().zip(&offset[x * size..(x + 1) * size]) {
            let sum = *o + d;
            *o = if sum >= p { sum - p } else { sum };
        }
        x = parent[x] as usize;
    }
    x
}

/// `(c, s)` of a part: the components of its graph and the dimension of the span of the twists
/// of its closed walks. A weighted union-find keeps each vertex's value relative to its root
/// under the edges `x(u) − y(v) = vec(u vᵀ)`. An edge inside a component closes a walk, and what
/// it misses by is that walk's twist up to sign. O(|part| · k l · log n + s · k l) time.
pub fn graph_count(t: &Table, part: &[u32], w: &mut Workspace) -> (usize, usize) {
    let (n, size, p) = (t.lines(), t.couplings(), t.p);
    for (i, parent) in w.parent.iter_mut().enumerate() {
        *parent = i as u32;
    }
    w.weight.fill(1);
    w.offset.fill(0);
    w.twists.clear();
    let mut components = n;
    for &cell in part {
        let cell = cell as usize;
        let (row, column) = (cell / t.columns, t.rows + cell % t.columns);
        let root_row = locate(&w.parent, &w.offset, p, row, &mut w.left);
        let root_column = locate(&w.parent, &w.offset, p, column, &mut w.right);
        let product = t.product(cell);
        // value(row) − value(column) − product, or for a new link, the root's offset up to sign.
        for e in 0..size {
            let target = if root_row == root_column {
                u16::from(w.left[e]) + u16::from(p - w.right[e]) + u16::from(p - product[e])
            } else {
                u16::from(product[e]) + u16::from(w.right[e]) + u16::from(p - w.left[e])
            };
            w.twist[e] = (target % u16::from(p)) as u8;
        }
        if root_row == root_column {
            if w.twist.iter().any(|&x| x != 0) {
                w.twists.insert(&mut w.twist);
            }
            continue;
        }
        let (child, root, negate) = if w.weight[root_row] <= w.weight[root_column] {
            (root_row, root_column, false)
        } else {
            (root_column, root_row, true)
        };
        w.parent[child] = root as u32;
        w.weight[root] += w.weight[child];
        for e in 0..size {
            let d = w.twist[e];
            w.offset[child * size + e] = if negate && d != 0 { p - d } else { d };
        }
        components -= 1;
    }
    (components, w.twists.rank)
}

/// The least number of nonzero cells of a nonzero function of the class, over all `p^dim` of
/// them. An odometer on the coefficients of a basis adds one basis function per step.
/// O(p^dim · cells) time, O(dim · cells) memory.
pub fn min_support(t: &Table) -> usize {
    let (cells, dim, p) = (t.cells(), t.dim(), t.p);
    let mut basis = vec![0u8; dim * cells];
    for cell in 0..cells {
        let (u, v) = (cell / t.columns, cell % t.columns);
        basis[u * cells + cell] = 1;
        if v > 0 {
            basis[(t.rows + v - 1) * cells + cell] = 1;
        }
        for (e, &m) in t.product(cell).iter().enumerate() {
            basis[(t.lines() - 1 + e) * cells + cell] = m;
        }
    }
    let mut digits = vec![0u8; dim];
    let mut f = vec![0u8; cells];
    let mut best = cells;
    loop {
        let mut i = 0;
        loop {
            if i == dim {
                return best;
            }
            for (y, &z) in f.iter_mut().zip(&basis[i * cells..(i + 1) * cells]) {
                let sum = *y + z;
                *y = if sum >= p { sum - p } else { sum };
            }
            digits[i] += 1;
            if digits[i] < p {
                break;
            }
            digits[i] = 0;
            i += 1;
        }
        best = best.min(f.iter().filter(|&&x| x != 0).count());
    }
}

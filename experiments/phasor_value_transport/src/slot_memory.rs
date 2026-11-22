//! Dense value memory and learned gates for the triple-transport read.
//! Its addresses are nearest directions in a trainable, unordered dictionary.
use crate::{
    Config, architecture,
    random::Random,
    slot_orbits::{self, ABSENT},
};

pub struct Layout {
    pub d: usize,
    pub slots: usize,
    pub dictionary: usize,
    pub values: usize,
    pub gates: usize,
    pub bias: usize,
}

pub fn capacity(head: crate::head::Head, width: usize, vocab: usize) -> usize {
    if head == crate::head::Head::LookupTripleTransport {
        vocab.min(256)
    } else {
        width.min(128)
    }
}

pub fn parameter_count(d: usize, k: usize) -> usize {
    4 * d + (k + k * k + 4) * d + 4
}

impl Layout {
    pub fn new(cfg: &Config) -> Self {
        let d = cfg.width;
        let slots = capacity(cfg.head, d, cfg.vocab);
        let dictionary = 0;
        let values = dictionary + slots * d;
        let gates = values + slots * slots * d;
        Self {
            d,
            slots,
            dictionary,
            values,
            gates,
            bias: gates + 4 * d,
        }
    }

    pub fn prototype<'a>(&self, w: &'a [f32], slot: usize) -> &'a [f32] {
        let start = self.dictionary + slot * self.d;
        &w[start..start + self.d]
    }

    pub fn row<'a>(&self, w: &'a [f32], cell: usize) -> &'a [f32] {
        let start = self.values + cell * self.d;
        &w[start..start + self.d]
    }
}

fn vector_order(a: &[f32], b: &[f32]) -> std::cmp::Ordering {
    a.iter()
        .zip(b)
        .map(|(x, y)| x.total_cmp(y))
        .find(|o| !o.is_eq())
        .unwrap_or(std::cmp::Ordering::Equal)
}

pub fn initialize(cfg: &Config, seed: u32, w: &mut [f32]) {
    let l = Layout::new(cfg);
    let mut rng = Random::new(seed.wrapping_add(30_000));
    let mut directions: Vec<_> = w[..cfg.vocab * l.d]
        .chunks_exact(l.d)
        .map(<[f32]>::to_vec)
        .collect();
    directions.sort_by(|a, b| vector_order(a, b));
    directions.truncate(l.slots);
    while directions.len() < l.slots {
        directions.push((0..l.d).map(|_| rng.gauss(0., 0.02) as f32).collect());
    }
    for a in &mut directions {
        let scale = (l.d as f32).sqrt() * inverse_norm(a);
        for x in a {
            *x *= scale;
        }
    }
    directions.sort_by(|a, b| vector_order(a, b));
    let w = &mut w[architecture::memory_start(cfg)..];
    for (dst, a) in w[l.dictionary..l.values]
        .chunks_exact_mut(l.d)
        .zip(directions)
    {
        dst.copy_from_slice(&a);
    }
    w[l.values..l.bias + 4].fill(0.);
}

pub fn dot(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

fn inverse_norm(v: &[f32]) -> f32 {
    let n = dot(v, v).sqrt();
    if n > 0. { n.recip() } else { 0. }
}

#[derive(Default)]
pub struct Routing {
    bank: Vec<u32>,
    priority: Vec<usize>,
    reads: Option<slot_orbits::Reads>,
    graph: slot_orbits::TripleGraph,
    inverse: Vec<f32>,
}

impl Routing {
    pub fn inverse_norms(&self) -> &[f32] {
        &self.inverse
    }

    // Products are hidden value · hidden prototype, computed by Kernite.
    pub fn refresh(&mut self, l: &Layout, w: &[f32], products: &[f32]) {
        self.inverse = (0..l.slots)
            .map(|i| inverse_norm(l.prototype(w, i)))
            .collect();
        let mut priority: Vec<_> = (0..l.slots).collect();
        priority.sort_by(|&a, &b| vector_order(l.prototype(w, a), l.prototype(w, b)));
        // Coincident prototypes have no content-based priority. Their cells
        // cannot be anchors, including when a tie straddles the anchor limit.
        let mut distinct = vec![true; l.slots];
        for pair in priority.windows(2) {
            if l.prototype(w, pair[0]) == l.prototype(w, pair[1]) {
                distinct[pair[0]] = false;
                distinct[pair[1]] = false;
            }
        }
        let mut bank = vec![ABSENT; l.slots * l.slots];
        for (cell, scores) in products.chunks_exact(l.slots).enumerate() {
            if !distinct[cell / l.slots] || !distinct[cell % l.slots] {
                continue;
            }
            let v = l.row(w, cell);
            let energy = dot(v, v);
            let damping = l.d as f32 * f32::EPSILON;
            if energy <= damping {
                continue;
            }
            let norm = (energy + damping).sqrt().recip();
            let mut best = std::f32::consts::FRAC_1_SQRT_2;
            for (i, &score) in scores.iter().enumerate() {
                let cosine = score * self.inverse[i] * norm;
                if cosine > best {
                    best = cosine;
                    bank[cell] = i as u32;
                } else if cosine == best {
                    bank[cell] = ABSENT;
                }
            }
        }
        if self.bank != bank || self.priority != priority {
            self.reads = Some(self.graph.complete(l.slots, &bank, &priority));
            self.bank = bank;
            self.priority = priority;
        }
    }

    fn nearest(&self, l: &Layout, w: &[f32], v: &[f32]) -> Option<usize> {
        let mut best = 0.;
        let mut winner = None;
        for (i, &inv) in self.inverse.iter().enumerate() {
            let similarity = dot(v, l.prototype(w, i)) * inv;
            if similarity > best {
                best = similarity;
                winner = Some(i);
            } else if similarity == best {
                winner = None;
            }
        }
        winner
    }
}

#[derive(Clone, Default)]
pub struct Row {
    pub cell: Option<usize>,
    pub constant: Option<usize>,
    pub transport: Option<usize>,
    pub coefficients: [f32; 4],
}

pub fn scales(d: usize) -> [f32; 4] {
    let root = (d as f32).sqrt();
    [root.recip(), d as f32, root, root]
}

pub fn bases(d: usize) -> [f32; 4] {
    let mut b = scales(d);
    b[2..].fill(0.);
    b
}

impl Row {
    pub fn read<'a>(
        &self,
        branch: usize,
        l: &Layout,
        w: &'a [f32],
        h: &'a [f32],
    ) -> Option<&'a [f32]> {
        match branch {
            0 => Some(h),
            1 => self.cell.map(|i| l.row(w, i)),
            2 => self.constant.map(|i| l.prototype(w, i)),
            3 => self.transport.map(|i| l.prototype(w, i)),
            _ => unreachable!(),
        }
    }
}

/// Independent scalar scan used to verify batched device addressing.
pub fn scalar_cells(
    l: &Layout,
    w: &[f32],
    route: &Routing,
    x: &[f32],
    hidden: &[f32],
    length: usize,
) -> Vec<u32> {
    let d = l.d;
    let units = (d as f32).sqrt().recip();
    let mut key = vec![0.; d];
    let mut prior = None;
    let mut cells = Vec::with_capacity(x.len() / d);
    for (r, (q, h)) in x.chunks_exact(d).zip(hidden.chunks_exact(d)).enumerate() {
        let cell = if r.is_multiple_of(length) {
            None
        } else {
            prior
                .zip(route.nearest(l, w, q))
                .map(|(a, b)| (a * l.slots + b) as u32)
        };
        cells.push(cell.unwrap_or(ABSENT));
        if !(r + 1).is_multiple_of(length) {
            for j in 0..d {
                key[j] = q[j] + units * h[j];
            }
            prior = route.nearest(l, w, &key);
        }
    }
    cells
}

pub fn forward(
    l: &Layout,
    w: &[f32],
    route: &Routing,
    cells: &[u32],
    hidden: &[f32],
    length: usize,
) -> (Vec<Row>, Vec<f32>) {
    let d = l.d;
    assert_eq!(cells.len() * d, hidden.len());
    let scale = scales(d);
    let units = (d as f32).sqrt().recip();
    let mut tape = Vec::with_capacity(cells.len());
    let mut out = vec![0.; hidden.len()];
    for (r, ((&cell, h), dst)) in cells
        .iter()
        .zip(hidden.chunks_exact(d))
        .zip(out.chunks_exact_mut(d))
        .enumerate()
    {
        let mut row = Row {
            coefficients: bases(d),
            ..Row::default()
        };
        for (i, g) in row.coefficients.iter_mut().enumerate() {
            let mut value = w[l.bias + i];
            if !r.is_multiple_of(length) {
                let weight = l.gates + i * d;
                value += units * dot(&w[weight..weight + d], &hidden[(r - 1) * d..r * d]);
            }
            *g += scale[i] * value;
        }
        if cell != ABSENT {
            let cell = cell as usize;
            let reads = route.reads.as_ref().expect("routing was refreshed");
            row.cell = Some(cell);
            row.constant =
                (reads.constant[cell] != ABSENT).then_some(reads.constant[cell] as usize);
            row.transport =
                (reads.transport[cell] != ABSENT).then_some(reads.transport[cell] as usize);
        }
        for (i, &c) in row.coefficients.iter().enumerate() {
            if let Some(value) = row.read(i, l, w, h) {
                for (y, &v) in dst.iter_mut().zip(value) {
                    *y += c * v;
                }
            }
        }
        tape.push(row);
    }
    (tape, out)
}

pub fn backward(
    l: &Layout,
    w: &[f32],
    hidden: &[f32],
    rows: &[Row],
    dy: &[f32],
    length: usize,
    dw: &mut [f32],
) -> Vec<f32> {
    let d = l.d;
    let scale = scales(d);
    let units = (d as f32).sqrt().recip();
    let mut dh = vec![0.; rows.len() * d];
    dw[l.dictionary..l.bias + 4].fill(0.);
    for (r, row) in rows.iter().enumerate() {
        let v = &dy[r * d..(r + 1) * d];
        let h = &hidden[r * d..(r + 1) * d];
        for (i, &c) in row.coefficients.iter().enumerate() {
            let Some(value) = row.read(i, l, w, h) else {
                continue;
            };
            let dg = scale[i] * dot(value, v);
            dw[l.bias + i] += dg;
            if !r.is_multiple_of(length) {
                for j in 0..d {
                    let k = l.gates + i * d + j;
                    dw[k] += units * dg * hidden[(r - 1) * d + j];
                    dh[(r - 1) * d + j] += units * dg * w[k];
                }
            }
            let dst = match i {
                0 => &mut dh[r * d..(r + 1) * d],
                1 => &mut dw[l.values + row.cell.unwrap() * d..][..d],
                2 => &mut dw[l.dictionary + row.constant.unwrap() * d..][..d],
                3 => &mut dw[l.dictionary + row.transport.unwrap() * d..][..d],
                _ => unreachable!(),
            };
            for (g, &z) in dst.iter_mut().zip(v) {
                *g += c * z;
            }
        }
    }
    dh
}

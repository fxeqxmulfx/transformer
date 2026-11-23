//! Append-only soft scalar-address reads with a learned per-write log prior.
//! AddressGap proves the planar score equivalence and the soft value bound.
use super::index_audit;
use super::planar::{Index, Line};
use super::{add, dot};
use std::rc::Rc;
use std::{hint::black_box, time::Instant};

struct Entry {
    addr: f64,
    log_gate: f64,
    payload: Vec<f32>,
}
struct Adjoint {
    coord: f64,
    score: f64,
    value: Vec<f32>,
}
pub(super) struct Cells {
    entries: Vec<Rc<Entry>>,
    adjoints: Vec<Adjoint>,
    width: usize,
    hull: Option<Index>,
    hard: bool,
    build_ns: u128,
}
impl Cells {
    pub fn new(width: usize) -> Self {
        Self {
            entries: Vec::new(),
            adjoints: Vec::new(),
            width,
            hull: index_audit::enabled().then(Index::default),
            hard: index_audit::hard(),
            build_ns: 0,
        }
    }
    pub fn reset(&mut self) {
        self.entries.clear();
        if let Some(index) = &mut self.hull {
            index.clear();
        }
        self.build_ns = 0;
        self.adjoints.clear();
    }
    pub fn time_last(&self, coord: f32, quality: Option<[f64; 4]>, soft_ns: u128) {
        let (Some(idx), Some(qinfo)) = (&self.hull, quality) else {
            return;
        };
        let q = f64::from(coord);
        let repetitions = 16;
        let start = Instant::now();
        let mut indexed = usize::MAX;
        for _ in 0..repetitions {
            indexed = black_box(idx.best(black_box(q)).unwrap().index);
        }
        let index = start.elapsed().as_nanos() / repetitions;
        let start = Instant::now();
        let mut scanned = usize::MAX;
        for _ in 0..repetitions {
            let x = black_box(q);
            let mut high = f64::NEG_INFINITY;
            for (i, e) in self.entries.iter().enumerate() {
                let score = e.log_gate - (x - e.addr).powi(2);
                if score >= high {
                    high = score;
                    scanned = i;
                }
            }
            black_box(scanned);
        }
        let scan = start.elapsed().as_nanos() / repetitions;
        index_audit::observe(index_audit::Timing {
            writes: self.entries.len(),
            disagrees: indexed != scanned,
            sharp: qinfo[1] >= ((qinfo[0] - 1.) * 100.).ln(),
            index,
            scan,
            soft: soft_ns,
            build: self.build_ns,
        });
    }
    fn ensure(&mut self, k: usize) {
        if self.adjoints.len() <= k {
            self.adjoints.resize_with(k + 1, || Adjoint {
                coord: 0.,
                score: 0.,
                value: vec![0.; self.width],
            });
        }
    }
}
struct Donor {
    entry: Rc<Entry>,
    p: f64,
    match_strength: f64,
}
pub(super) struct Read {
    pub value: Vec<f32>,
    donors: Vec<Donor>,
    coord: f64,
    pub quality: Option<[f64; 4]>,
}
impl Read {
    pub fn at(cells: &Cells, coord: f32) -> Self {
        let q = f64::from(coord);
        let dim = cells.width - 1;
        let mut read = Self {
            value: vec![0.; cells.width],
            donors: Vec::new(),
            coord: q,
            quality: None,
        };
        if !q.is_finite() || cells.entries.is_empty() {
            return read;
        }
        if cells.hard {
            let line = cells.hull.as_ref().unwrap().best(q).unwrap();
            read.value[..dim].copy_from_slice(&cells.entries[line.index].payload);
            read.value[dim] = line.score(q).exp() as f32;
            return read;
        }
        let mut scores = Vec::with_capacity(cells.entries.len());
        let (mut best, mut high, mut second) = (0, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for (i, entry) in cells.entries.iter().enumerate() {
            let score = entry.log_gate - (q - entry.addr).powi(2);
            if score >= high {
                second = high;
                high = score;
                best = i;
            } else if score > second {
                second = score;
            }
            scores.push(score);
        }
        let denom = scores.iter().map(|x| (x - high).exp()).sum::<f64>();
        let top_mass = denom.recip();
        for (entry, score) in cells.entries.iter().zip(scores) {
            let p = (score - high).exp() / denom;
            let strength = score.exp();
            add(&mut read.value[..dim], &entry.payload, p as f32);
            read.value[dim] += (p * strength) as f32;
            read.donors.push(Donor {
                entry: Rc::clone(entry),
                p,
                match_strength: strength,
            });
        }
        let rms = (read.value[..dim]
            .iter()
            .zip(&cells.entries[best].payload)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / dim as f64)
            .sqrt();
        if cells.entries.len() >= 2 {
            read.quality = Some([cells.entries.len() as f64, high - second, top_mass, rms]);
        }
        read
    }
    pub fn backward(&self, mem: &mut Cells, g: &[f32]) -> f32 {
        let d = mem.width - 1;
        let expected = f64::from(dot(&self.value, g));
        let mut dq = 0.;
        for (i, v) in self.donors.iter().enumerate() {
            mem.ensure(i);
            let a = &mut mem.adjoints[i];
            let projected = f64::from(dot(&v.entry.payload, &g[..d]));
            let ds = v.p * (projected + 2. * f64::from(g[d]) * v.match_strength - expected);
            let da = 2. * (self.coord - v.entry.addr) * ds;
            a.coord += da;
            a.score += ds;
            add(&mut a.value[..d], &g[..d], v.p as f32);
            dq -= da;
        }
        dq as f32
    }
}
pub(super) struct Write {
    index: usize,
    gate: f32,
}
impl Write {
    pub fn apply(mem: &mut Cells, coord: f32, target: &[f32], gate: f32) -> Option<Self> {
        if !coord.is_finite() || gate <= 0. {
            return None;
        }
        let at = mem.entries.len();
        if let Some(hull) = &mut mem.hull {
            let start = Instant::now();
            hull.write(Line {
                address: f64::from(coord),
                log_weight: f64::from(gate).ln(),
                index: at,
            });
            mem.build_ns += start.elapsed().as_nanos();
        }
        mem.entries.push(Rc::new(Entry {
            addr: f64::from(coord),
            log_gate: f64::from(gate).ln(),
            payload: target[..mem.width - 1].to_vec(),
        }));
        Some(Self { index: at, gate })
    }
    pub fn backward(&self, mem: &mut Cells) -> (f32, f32, Vec<f32>) {
        mem.ensure(self.index);
        let a = &mut mem.adjoints[self.index];
        (
            a.coord as f32,
            (a.score * (1. - f64::from(self.gate))) as f32,
            std::mem::take(&mut a.value),
        )
    }
}

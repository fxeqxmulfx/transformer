//! Causal association store. The middle embedding of every consecutive
//! triple names a relation; both orientations share the same storage rules.
use std::collections::BTreeMap;

pub(super) const BINS: usize = 1 << 15;

#[derive(Clone, Copy)]
struct Edge {
    dest: usize,
    position: usize,
    written_at: usize,
}
struct Bindings {
    relation: usize,
    writes: Writes,
}
enum Writes {
    Latest(Edge),
    Ordered(Vec<Edge>),
}
impl Writes {
    fn add(&mut self, edge: Edge) {
        match self {
            Self::Latest(last) => *last = edge,
            Self::Ordered(history) => history.push(edge),
        }
    }
    fn read(&self, previous: Option<usize>) -> Option<&Edge> {
        match self {
            Self::Latest(edge) => Some(edge),
            Self::Ordered(history) => {
                let next = previous.map_or(0, |t| history.partition_point(|e| e.written_at <= t));
                history.get(next)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Witness {
    pub position: usize,
    code: u32,
}

impl Witness {
    #[cfg(test)]
    pub(super) fn at_code(code: u32) -> Self {
        Self { position: 0, code }
    }

    /// The low and high halves independently address disjoint banks. Each
    /// half uses fourteen address bits and its own sign bit.
    pub(super) fn addresses(self) -> [(usize, f64); 2] {
        let half = BINS / 2;
        let slot = |bits: u32| bits as usize & (half - 1);
        let sign = |bits: u32| if bits & (1 << 15) == 0 { 1. } else { -1. };
        let high = self.code >> 16;
        [
            (slot(self.code), sign(self.code)),
            (half + slot(high), sign(high)),
        ]
    }

    pub(super) fn weight(self) -> (usize, f32) {
        (
            (self.code as usize) % BINS,
            if self.code & BINS as u32 == 0 {
                1.
            } else {
                -1.
            },
        )
    }

    pub(super) fn vote(self, votes: &mut [f32], table: &[f32]) {
        let (bin, sign) = self.weight();
        votes[self.position] += sign * table[bin];
    }
}

// Signed feature hashing bounds the number of learned scalar weights.
// The key consists solely of three geometric addresses and two directions.
pub(super) fn code(cue: usize, relation: usize, terminal: usize) -> u32 {
    let mut h = 0x9e37_79b9u32;
    for v in [cue, relation, terminal] {
        h ^= (v as u32).wrapping_mul(0x85eb_ca6b);
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

#[derive(Default)]
pub(super) struct Store {
    ordered: bool,
    cues: BTreeMap<usize, usize>,
    outgoing: BTreeMap<usize, Vec<Bindings>>,
    previous: [Option<usize>; 2],
}
impl Store {
    pub fn new(ordered: bool) -> Self {
        Self {
            ordered,
            ..Self::default()
        }
    }
    fn write(&mut self, source: usize, relation: usize, edge: Edge) {
        let list = self.outgoing.entry(source).or_default();
        if let Some(old) = list.iter_mut().find(|p| p.relation == relation) {
            old.writes.add(edge);
        } else {
            list.push(Bindings {
                relation,
                writes: if self.ordered {
                    Writes::Ordered(vec![edge])
                } else {
                    Writes::Latest(edge)
                },
            });
        }
    }
    pub fn observe(&mut self, current: Option<usize>, t: usize) {
        let [left, middle] = self.previous;
        if let (Some(cue), Some(dest)) = (middle, current) {
            self.cues.insert(cue, dest);
        }
        if let (Some(a), Some(r), Some(b)) = (left, middle, current) {
            self.write(
                a,
                2 * r,
                Edge {
                    dest: b,
                    position: t,
                    written_at: t,
                },
            );
            self.write(
                b,
                2 * r + 1,
                Edge {
                    dest: a,
                    position: t - 2,
                    written_at: t,
                },
            );
        }
        self.previous = [middle, current];
    }
    fn endpoints(&self, first: &Bindings, budgets: &[usize], mut sample: impl FnMut(usize, Edge)) {
        let mut edge = *first.writes.read(None).expect("empty relation history");
        let mut depth = 1;
        let mut finished = false;
        for (i, &budget) in budgets.iter().enumerate() {
            while !finished && depth < budget {
                let next = self
                    .outgoing
                    .get(&edge.dest)
                    .and_then(|rels| rels.iter().find(|x| x.relation == first.relation))
                    .and_then(|xs| xs.writes.read(Some(edge.written_at)));
                match next {
                    Some(next) if self.ordered || next.dest != edge.dest => {
                        edge = *next;
                        depth += 1;
                    }
                    _ => finished = true,
                }
            }
            sample(i, edge);
        }
    }
    pub(super) fn visit(&self, passes: usize, mut emit: impl FnMut(Witness)) {
        self.visit_budgets(&[passes], |_, witness| emit(witness));
    }

    /// Traverse each cue and relation once, sampling in increasing budget order.
    /// Within each budget the floating-point vote order is the single-read order.
    pub(super) fn visit_budgets(&self, budgets: &[usize], mut emit: impl FnMut(usize, Witness)) {
        self.visit_roles(budgets, |i, witness, _| emit(i, witness));
    }

    pub(super) fn visit_roles(
        &self,
        budgets: &[usize],
        mut emit: impl FnMut(usize, Witness, [usize; 3]),
    ) {
        debug_assert!(budgets.first().is_none_or(|&n| n > 0) && budgets.is_sorted());
        for (&cue, root) in &self.cues {
            let Some(firsts) = self.outgoing.get(root) else {
                continue;
            };
            for first in firsts {
                self.endpoints(first, budgets, |i, end| {
                    if let Some(tails) = self.outgoing.get(&end.dest) {
                        for last in tails {
                            if let Some(record) = last.writes.read(Some(end.written_at)) {
                                emit(
                                    i,
                                    Witness {
                                        position: record.position,
                                        code: code(cue, first.relation, last.relation),
                                    },
                                    [cue, first.relation, last.relation],
                                );
                            }
                        }
                    }
                });
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn self_links_rebindings_and_reverse_positions_advance_by_write_time() {
        let mut m = Store::new(true);
        for (i, (a, b)) in [(0, 0), (0, 1), (1, 0), (0, 2)].into_iter().enumerate() {
            m.write(
                a,
                1,
                Edge {
                    dest: b,
                    position: 2 * i,
                    written_at: 2 * i + 2,
                },
            );
        }
        let first = &m.outgoing[&0][0];
        let mut observed = Vec::new();
        m.endpoints(first, &[1, 2, 3, 40], |_, e| {
            observed.push((e.dest, e.written_at))
        });
        assert_eq!(observed, [(0, 2), (1, 4), (0, 6), (2, 8)]);
        let last = &m.outgoing[&0][0];
        assert_eq!(last.writes.read(Some(2)).unwrap().position, 2);
        assert!(last.writes.read(Some(8)).is_none());
    }
}

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
    fn endpoint(&self, first: &Bindings, passes: usize) -> Edge {
        let mut edge = *first.writes.read(None).expect("empty relation history");
        for _ in 1..passes {
            let next = self
                .outgoing
                .get(&edge.dest)
                .and_then(|rels| rels.iter().find(|x| x.relation == first.relation))
                .and_then(|xs| xs.writes.read(Some(edge.written_at)));
            let Some(next) = next else {
                break;
            };
            if !self.ordered && next.dest == edge.dest {
                break;
            }
            edge = *next;
        }
        edge
    }
    pub(super) fn visit(&self, passes: usize, mut emit: impl FnMut(Witness)) {
        for (&cue, root) in &self.cues {
            let Some(firsts) = self.outgoing.get(root) else {
                continue;
            };
            for first in firsts {
                let end = self.endpoint(first, passes);
                if let Some(tails) = self.outgoing.get(&end.dest) {
                    for last in tails {
                        if let Some(record) = last.writes.read(Some(end.written_at)) {
                            emit(Witness {
                                position: record.position,
                                code: code(cue, first.relation, last.relation),
                            });
                        }
                    }
                }
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
        assert_eq!(
            (m.endpoint(first, 1).dest, m.endpoint(first, 2).dest),
            (0, 1)
        );
        assert_eq!(m.endpoint(first, 3).dest, 0);
        assert_eq!(
            (m.endpoint(first, 40).dest, m.endpoint(first, 40).written_at),
            (2, 8)
        );
        let last = &m.outgoing[&0][0];
        assert_eq!(last.writes.read(Some(2)).unwrap().position, 2);
        assert!(last.writes.read(Some(8)).is_none());
    }
}

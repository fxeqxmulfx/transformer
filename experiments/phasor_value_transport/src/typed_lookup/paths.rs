//! Causal association store. The middle embedding of every consecutive
//! triple names a relation; both orientations share the same storage rules.
use std::collections::BTreeMap;

pub(super) const BINS: usize = 1 << 15;

#[derive(Clone, Copy)]
struct Edge {
    relation: usize,
    dest: usize,
    position: usize,
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
    cues: BTreeMap<usize, usize>,
    outgoing: BTreeMap<usize, Vec<Edge>>,
    previous: [Option<usize>; 2],
}

impl Store {
    fn write(&mut self, source: usize, edge: Edge) {
        let list = self.outgoing.entry(source).or_default();
        if let Some(old) = list.iter_mut().find(|p| p.relation == edge.relation) {
            *old = edge;
        } else {
            list.push(edge);
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
                Edge {
                    relation: 2 * r,
                    dest: b,
                    position: t,
                },
            );
            self.write(
                b,
                Edge {
                    relation: 2 * r + 1,
                    dest: a,
                    position: t - 2,
                },
            );
        }
        self.previous = [middle, current];
    }

    pub(super) fn visit(&self, passes: usize, mut emit: impl FnMut(Witness)) {
        for (&cue, root) in &self.cues {
            let Some(firsts) = self.outgoing.get(root) else {
                continue;
            };
            for first in firsts {
                let mut key = first.dest;
                for _ in 1..passes {
                    let Some(later) = self
                        .outgoing
                        .get(&key)
                        .and_then(|xs| xs.iter().find(|x| x.relation == first.relation))
                    else {
                        break;
                    };
                    if later.dest == key {
                        break;
                    }
                    key = later.dest;
                }
                if let Some(tails) = self.outgoing.get(&key) {
                    for last in tails {
                        emit(Witness {
                            position: last.position,
                            code: code(cue, first.relation, last.relation),
                        });
                    }
                }
            }
        }
    }
}

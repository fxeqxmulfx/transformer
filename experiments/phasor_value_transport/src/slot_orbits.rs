//! Equivalence closure on ordered triples of unnamed prototype slots.
//! The caller supplies partial maps extracted from learned hidden vectors.
//! Retained edges follow TripleTransport.connected_growing_bank.
#[path = "slot_candidates.rs"]
mod candidates;
use candidates::RowCandidates;

pub const ABSENT: u32 = u32::MAX;

struct Components {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl Components {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len as u32).collect(),
            rank: vec![0; len],
        }
    }

    fn root(&mut self, mut n: usize) -> usize {
        while self.parent[n] as usize != n {
            let next = self.parent[n] as usize;
            self.parent[n] = self.parent[next];
            n = self.parent[n] as usize;
        }
        n
    }

    fn join(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.root(a), self.root(b));
        if a == b {
            return;
        }
        if self.rank[a] < self.rank[b] {
            std::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a as u32;
        self.rank[a] += u8::from(self.rank[a] == self.rank[b]);
    }

    fn flatten(&mut self) {
        for i in 0..self.parent.len() {
            self.parent[i] = self.root(i) as u32;
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reads {
    pub constant: Vec<u32>,
    pub transport: Vec<u32>,
}

type RowMap = (usize, Vec<(usize, usize)>);

/// The closure of the partial injective maps having at least three matches.
/// Slot order only chooses the anchors at read time.
#[derive(Default)]
pub struct TripleGraph {
    separate_images: bool,
    slots: usize,
    supported: Vec<u32>,
    components: Option<Components>,
}

impl TripleGraph {
    pub fn new(separate_images: bool) -> Self {
        Self {
            separate_images,
            ..Self::default()
        }
    }

    fn image_offset(&self, slots: usize) -> usize {
        if self.separate_images {
            slots.pow(3)
        } else {
            0
        }
    }

    fn update(&mut self, slots: usize, bank: &[u32], maps: &[RowMap]) -> Option<&[u32]> {
        if maps.iter().all(|(_, points)| points.len() < 3) {
            self.components = None;
            self.supported.clear();
            return None;
        }
        let mut supported = vec![ABSENT; bank.len()];
        for (r, points) in maps {
            if points.len() >= 3 {
                let row = r * slots..(r + 1) * slots;
                supported[row.clone()].copy_from_slice(&bank[row]);
            }
        }
        if self.slots != slots || self.supported != supported {
            let growing = self.slots == slots
                && self.supported.len() == supported.len()
                && self
                    .supported
                    .iter()
                    .zip(&supported)
                    .all(|(&old, &new)| old == ABSENT || old == new);
            if !growing {
                self.components = None;
                self.supported = vec![ABSENT; supported.len()];
            }
            let role_size = slots.pow(3);
            let image_offset = self.image_offset(slots);
            let triple = |a, b, c| (a * slots + b) * slots + c;
            for (r, points) in maps.iter().filter(|(_, p)| p.len() >= 3) {
                let old = &self.supported[r * slots..(r + 1) * slots];
                let fresh: Vec<_> = points
                    .iter()
                    .copied()
                    .filter(|&(x, _)| old[x] == ABSENT)
                    .collect();
                if fresh.is_empty() {
                    continue;
                }
                let components = self
                    .components
                    .get_or_insert_with(|| Components::new(image_offset + role_size));
                for &(u, s) in points {
                    for &(v, t) in points {
                        if u == v {
                            continue;
                        }
                        // If both anchors already existed, only a new third point adds an edge.
                        let thirds = if old[u] == ABSENT || old[v] == ABSENT {
                            points
                        } else {
                            &fresh
                        };
                        for &(x, y) in thirds {
                            if x != u && x != v {
                                components.join(triple(u, v, x), image_offset + triple(s, t, y));
                            }
                        }
                    }
                }
            }
            if let Some(components) = &mut self.components {
                components.flatten();
            }
            self.slots = slots;
            self.supported = supported;
        }
        self.components.as_ref().map(|c| c.parent.as_slice())
    }

    /// The same two output arrays serve every position and every sequence length.
    /// Priority follows the content of the prototype vectors and moves with them
    /// under slot renaming. Four distinct observed anchors supply up to six
    /// unordered anchor pairs. Each pair is used in its priority orientation.
    pub fn complete(&mut self, slots: usize, bank: &[u32], priority: &[usize]) -> Reads {
        assert!(slots > 0 && slots <= 256 && bank.len().is_multiple_of(slots));
        assert_eq!(priority.len(), slots);
        let mut reads = Reads {
            constant: vec![ABSENT; bank.len()],
            transport: vec![ABSENT; bank.len()],
        };
        let mut maps = Vec::new();
        for (r, row) in bank.chunks_exact(slots).enumerate() {
            let mut seen = vec![false; slots];
            let mut injective = true;
            let mut points = Vec::new();
            for &x in priority {
                let y = row[x];
                if y != ABSENT {
                    let y = y as usize;
                    injective &= !seen[y];
                    seen[y] = true;
                    points.push((x, y));
                }
            }
            if points.len() >= 2 && points.iter().all(|p| p.1 == points[0].1) {
                for x in 0..slots {
                    // The current query is never one of its own two supports.
                    if points.len() - usize::from(row[x] != ABSENT) >= 2 {
                        reads.constant[r * slots + x] = points[0].1 as u32;
                    }
                }
            } else if injective && points.len() >= 2 {
                maps.push((r, points));
            }
        }
        let image_offset = self.image_offset(slots);
        let Some(component) = self.update(slots, bank, &maps) else {
            return reads;
        };
        let pair = |x, y| (x * slots + y) * slots;
        for (r, points) in maps {
            let row = &bank[r * slots..(r + 1) * slots];
            let mut available = RowCandidates::new(slots, &points);
            for (x, output) in reads.transport[r * slots..(r + 1) * slots]
                .iter_mut()
                .enumerate()
            {
                let mut anchors = [(0, 0); 4];
                let mut count = 0;
                for &(u, s) in &points {
                    if u != x {
                        anchors[count] = (u, s);
                        count += 1;
                        if count == anchors.len() {
                            break;
                        }
                    }
                }
                let mut answer = None;
                'anchors: for i in 0..count {
                    let (u, s) = anchors[i];
                    for &(v, t) in &anchors[i + 1..count] {
                        let key = component[pair(u, v) + x];
                        let dst = image_offset + pair(s, t);
                        if let Some(y) =
                            available.unique(dst, &component[dst..dst + slots], key, row[x], (s, t))
                        {
                            if answer.is_some_and(|z| z != y) {
                                answer = None;
                                break 'anchors;
                            }
                            answer = Some(y);
                        }
                    }
                }
                if let Some(y) = answer {
                    *output = y;
                }
            }
        }
        reads
    }
}

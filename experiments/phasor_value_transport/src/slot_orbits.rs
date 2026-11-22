//! Equivalence closure on ordered triples of unnamed prototype slots.
//! The caller supplies partial maps extracted from learned hidden vectors.
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

    fn flatten(mut self) -> Vec<u32> {
        for i in 0..self.parent.len() {
            self.parent[i] = self.root(i) as u32;
        }
        self.parent
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reads {
    pub constant: Vec<u32>,
    pub transport: Vec<u32>,
}

/// The same two output arrays serve every position and every sequence length.
/// Priority follows the content of the prototype vectors and moves with them
/// under slot renaming. Four distinct observed anchors supply up to six
/// unordered anchor pairs. Each pair is used in its priority orientation.
pub fn complete(slots: usize, bank: &[u32], priority: &[usize]) -> Reads {
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
    if maps.iter().all(|(_, points)| points.len() < 3) {
        return reads;
    }
    let pair = |x, y| (x * slots + y) * slots;
    let mut components = Components::new(slots.pow(3));
    for (_, points) in &maps {
        for &(u, s) in points {
            for &(v, t) in points {
                if u != v {
                    let (src, dst) = (pair(u, v), pair(s, t));
                    for &(x, y) in points {
                        if x != u && x != v {
                            components.join(src + x, dst + y);
                        }
                    }
                }
            }
        }
    }
    let component = components.flatten();
    for (r, points) in maps {
        let row = &bank[r * slots..(r + 1) * slots];
        let mut owner = vec![usize::MAX; slots];
        for &(x, y) in &points {
            owner[y] = x;
        }
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
                    let dst = pair(s, t);
                    let mut candidates = (0..slots).filter(|&y| {
                        y != s
                            && y != t
                            && (owner[y] == usize::MAX || row[x] == y as u32)
                            && component[dst + y] == key
                    });
                    if let Some(y) = candidates.next()
                        && candidates.next().is_none()
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
                *output = y as u32;
            }
        }
    }
    reads
}

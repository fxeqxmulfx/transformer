//! Number of distinct sources with two composable edges for each relation.
//! The update follows Transformer.Phasor.continuingSources_update. The table
//! and its inverse counts contain only triples completed within one prompt.
use crate::engine::Result;
use std::collections::BTreeMap;

pub type Edges = BTreeMap<(u32, u32), u32>;

#[derive(Clone, Copy, Default)]
struct Size {
    sources: usize,
    destinations: usize,
}

pub struct Support {
    incoming: BTreeMap<(u32, u32), usize>,
    scores: Vec<usize>,
    sizes: Vec<Size>,
}

impl Support {
    pub fn new(vocab: usize) -> Self {
        Self {
            incoming: BTreeMap::new(),
            scores: vec![0; vocab],
            sizes: vec![Size::default(); vocab],
        }
    }

    pub fn clear(&mut self) {
        self.incoming.clear();
        self.scores.fill(0);
        self.sizes.fill(Size::default());
    }

    // Inspect the old table before replacing this edge. A new source makes
    // each previous edge into it composable, as well as any new self-loop.
    pub fn before_write(&mut self, edges: &Edges, source: u32, relation: u32, end: u32) {
        let key = (source, relation);
        let score = &mut self.scores[relation as usize];
        let size = &mut self.sizes[relation as usize];
        if let Some(&old) = edges.get(&key) {
            if old == end {
                return;
            }
            if edges.contains_key(&(old, relation)) {
                *score -= 1;
            }
            let old_count = self
                .incoming
                .get_mut(&(old, relation))
                .expect("every stored edge has an incoming count");
            *old_count -= 1;
            size.destinations -= usize::from(*old_count == 0);
        } else {
            size.sources += 1;
            *score += self.incoming.get(&key).copied().unwrap_or(0);
        }
        if end == source || edges.contains_key(&(end, relation)) {
            *score += 1;
        }
        let count = self.incoming.entry((end, relation)).or_default();
        size.destinations += usize::from(*count == 0);
        *count += 1;
    }

    pub fn retain_maximal(&self, endpoints: &mut [u32], source: u32) -> usize {
        let maximum = endpoints
            .iter()
            .zip(&self.scores)
            .filter_map(|(&end, &score)| (end != source).then_some(score))
            .max()
            .unwrap_or(0);
        let mut kept = 0;
        for (end, &score) in endpoints.iter_mut().zip(&self.scores) {
            if score < maximum {
                *end = source;
            } else if *end != source {
                kept += 1;
            }
        }
        kept
    }
}

/// One address from closed relations, as in Phasor.closedRelationAddress.
/// A tie between different endpoints leaves the ordinary successor in place.
pub struct ClosedRead {
    edges: Edges,
    support: Support,
}

impl ClosedRead {
    pub fn new(vocab: usize) -> Self {
        Self {
            edges: Edges::new(),
            support: Support::new(vocab),
        }
    }

    pub fn clear(&mut self) {
        self.edges.clear();
        self.support.clear();
    }

    pub fn write(&mut self, source: u32, relation: u32, end: u32) {
        self.support
            .before_write(&self.edges, source, relation, end);
        self.edges.insert((source, relation), end);
    }

    pub fn address(&self, source: u32) -> Option<u32> {
        let mut largest = 0;
        let mut answer = None;
        for (&(_, h), &middle) in self.edges.range((source, 0)..=(source, u32::MAX)) {
            let size = self.support.sizes[h as usize];
            if self.support.scores[h as usize] != size.sources || size.destinations != size.sources
            {
                continue;
            }
            let end = self.edges[&(middle, h)];
            if end == source {
                continue;
            }
            if size.sources > largest {
                largest = size.sources;
                answer = Some(end);
            } else if size.sources == largest && answer != Some(end) {
                answer = None;
            }
        }
        answer
    }
}

// Independently recount every two-edge source after each write on all
// eight-token prompts over three symbols, including resets and overwrites.
pub fn check_counts() -> Result<()> {
    let mut edges = Edges::new();
    let mut support = Support::new(3);
    for mut code in 0..3u32.pow(8) {
        let mut tokens = [0; 8];
        for token in &mut tokens {
            *token = code % 3;
            code /= 3;
        }
        edges.clear();
        support.clear();
        for t in tokens.windows(3) {
            support.before_write(&edges, t[0], t[1], t[2]);
            edges.insert((t[0], t[1]), t[2]);
            for h in 0..3 {
                let sources = (0..3).filter(|&a| edges.contains_key(&(a, h))).count();
                let destinations = (0..3)
                    .filter(|&b| (0..3).any(|a| edges.get(&(a, h)) == Some(&b)))
                    .count();
                let size = support.sizes[h as usize];
                if (size.sources, size.destinations) != (sources, destinations) {
                    return Err("relation domain or range differs from the edge recount".into());
                }
                let expected = (0..3)
                    .filter(|&a| {
                        edges
                            .get(&(a, h))
                            .and_then(|&b| edges.get(&(b, h)))
                            .is_some()
                    })
                    .count();
                if support.scores[h as usize] != expected {
                    return Err(
                        "streaming relation support differs from the two-edge recount".into(),
                    );
                }
            }
        }
    }
    Ok(())
}

//! Number of distinct sources with two composable edges for each relation.
//! The update follows Transformer.Phasor.continuingSources_update. The table
//! and its inverse counts contain only triples completed within one prompt.
use crate::engine::Result;
use std::collections::BTreeMap;

pub type Edges = BTreeMap<(u32, u32), u32>;

pub struct Support {
    incoming: BTreeMap<(u32, u32), usize>,
    scores: Vec<usize>,
}

impl Support {
    pub fn new(vocab: usize) -> Self {
        Self {
            incoming: BTreeMap::new(),
            scores: vec![0; vocab],
        }
    }

    pub fn clear(&mut self) {
        self.incoming.clear();
        self.scores.fill(0);
    }

    // Inspect the old table before replacing this edge. A new source makes
    // each previous edge into it composable, as well as any new self-loop.
    pub fn before_write(&mut self, edges: &Edges, source: u32, relation: u32, end: u32) {
        let key = (source, relation);
        let score = &mut self.scores[relation as usize];
        if let Some(&old) = edges.get(&key) {
            if old == end {
                return;
            }
            if edges.contains_key(&(old, relation)) {
                *score -= 1;
            }
            *self
                .incoming
                .get_mut(&(old, relation))
                .expect("every stored edge has an incoming count") -= 1;
        } else {
            *score += self.incoming.get(&key).copied().unwrap_or(0);
        }
        if end == source || edges.contains_key(&(end, relation)) {
            *score += 1;
        }
        *self.incoming.entry((end, relation)).or_default() += 1;
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

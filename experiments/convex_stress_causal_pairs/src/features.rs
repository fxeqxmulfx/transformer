//! CFt4 input maps. Presence precedes position-sensitive candidates by an
//! explicit prior, not an identification claim. No task name or label is read.
use crate::*;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feature {
    Presence(usize),
    Count(usize),
    Length,
    Lag(usize),
    Successor { token: usize, hop: usize },
}
pub fn library(vocabulary: usize, context: usize) -> Vec<Feature> {
    (0..vocabulary)
        .map(Feature::Presence)
        .chain((0..vocabulary).map(Feature::Count))
        .chain([Feature::Length])
        .chain((0..context).map(Feature::Lag))
        .chain(
            (1..=2).flat_map(|hop| {
                (0..vocabulary).map(move |token| Feature::Successor { token, hop })
            }),
        )
        .collect()
}
/// Single pass for counts and latest completed successors, with PAD=vocabulary.
pub fn encode(vocabulary: usize, context: usize, tokens: &[usize]) -> Vec<u32> {
    let mut counts = vec![0u32; vocabulary];
    let mut successor = vec![vocabulary as u32; 2 * vocabulary];
    for (i, &t) in tokens.iter().enumerate() {
        counts[t] += 1;
        for hop in 1..=2 {
            if let Some(&v) = tokens.get(i + hop) {
                successor[(hop - 1) * vocabulary + t] = v as u32;
            }
        }
    }
    counts
        .iter()
        .map(|&n| u32::from(n > 0))
        .chain(counts.iter().copied())
        .chain([tokens.len() as u32])
        .chain((0..context).map(|lag| {
            tokens
                .len()
                .checked_sub(lag + 1)
                .map_or(vocabulary as u32, |i| tokens[i] as u32)
        }))
        .chain(successor)
        .collect()
}
/// Independent scalar interpretation, used to check selected decoders and
/// every retained contradiction. It never consults the producer's feature cache.
pub fn value(feature: Feature, vocabulary: usize, tokens: &[usize]) -> u32 {
    match feature {
        Feature::Presence(t) => u32::from(tokens.contains(&t)),
        Feature::Count(t) => tokens.iter().filter(|&&x| x == t).count() as u32,
        Feature::Length => tokens.len() as u32,
        Feature::Lag(lag) => tokens
            .iter()
            .rev()
            .nth(lag)
            .map_or(vocabulary as u32, |&v| v as u32),
        Feature::Successor { token, hop } => tokens
            .windows(hop + 1)
            .rfind(|window| window[0] == token)
            .map_or(vocabulary as u32, |window| window[hop] as u32),
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Candidate {
    Single(usize),
    Pair(usize, usize),
}
impl Candidate {
    pub fn cached(self, row: &[u32]) -> [u32; 2] {
        match self {
            Self::Single(k) => [row[k], 0],
            Self::Pair(a, b) => [row[a], row[b]],
        }
    }
    pub fn original(self, library: &[Feature], vocabulary: usize, tokens: &[usize]) -> [u32; 2] {
        match self {
            Self::Single(k) => [value(library[k], vocabulary, tokens), 0],
            Self::Pair(a, b) => [
                value(library[a], vocabulary, tokens),
                value(library[b], vocabulary, tokens),
            ],
        }
    }
    pub fn valid(self, keys: usize) -> bool {
        match self {
            Self::Single(k) => k < keys,
            Self::Pair(a, b) => a < b && b < keys,
        }
    }
}
/// Singles, then lag pairs by increasing maximum lag, then all other pairs.
/// Pairs are generated lazily; every pair appears once and no answer is used.
pub fn catalogue(vocabulary: usize, context: usize) -> impl Iterator<Item = Candidate> {
    let keys = 4 * vocabulary + context + 1;
    let lag_start = 2 * vocabulary + 1;
    (0..keys)
        .map(Candidate::Single)
        .chain(
            (1..context).flat_map(move |b| {
                (0..b).map(move |a| Candidate::Pair(lag_start + a, lag_start + b))
            }),
        )
        .chain((0..keys).flat_map(move |a| {
            (a + 1..keys)
                .filter(move |&b| !(a >= lag_start && b < lag_start + context))
                .map(move |b| Candidate::Pair(a, b))
        }))
}

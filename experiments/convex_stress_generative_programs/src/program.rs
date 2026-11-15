//! Fixed positional/order-statistic summaries of input prefixes. Delimiters
//! and offsets are enumerated from observed input symbols, never task names
//! or labels. End/missing states have no designated raw decoder value.
use crate::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Order {
    Index,
    ReverseIndex,
    AscendingRank,
    DescendingRank,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Program {
    pub delimiter: usize,
    pub skip: usize,
    pub latest: bool,
    pub order: Order,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    pub state: usize,
    pub positions: u64,
    pub buffer_symbols: usize,
}
pub fn alphabet(input: &Input) -> Vec<usize> {
    input
        .observations
        .iter()
        .flat_map(|o| o.tokens.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
pub fn catalogue(alphabet: &[usize], vocabulary: usize) -> Result<Vec<Program>> {
    if !(2..=126).contains(&vocabulary)
        || alphabet.is_empty()
        || alphabet.len() > vocabulary
        || alphabet.iter().any(|&t| t >= vocabulary)
        || alphabet.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("generative catalogue requires a sorted alphabet in2..126 raw classes".into());
    }
    let mut out = vec![];
    for &delimiter in alphabet {
        for skip in 0..=3 {
            for latest in [false, true] {
                for order in [
                    Order::Index,
                    Order::ReverseIndex,
                    Order::AscendingRank,
                    Order::DescendingRank,
                ] {
                    out.push(Program {
                        delimiter,
                        skip,
                        latest,
                        order,
                    });
                }
            }
        }
    }
    Ok(out)
}
impl Program {
    pub fn run(&self, tokens: &[usize]) -> Value {
        let end = if self.latest {
            tokens.iter().rposition(|&t| t == self.delimiter)
        } else {
            tokens.iter().position(|&t| t == self.delimiter)
        };
        let Some(end) = end.filter(|&i| i >= self.skip) else {
            return Value {
                state: STATES - 1,
                positions: tokens.len() as u64,
                buffer_symbols: 0,
            };
        };
        let mut word = tokens[self.skip..end].to_vec();
        let positions = (tokens.len() + word.len()) as u64;
        let buffer_symbols = word.len();
        match self.order {
            Order::Index => {}
            Order::ReverseIndex => word.reverse(),
            Order::AscendingRank => word.sort_unstable(),
            Order::DescendingRank => word.sort_unstable_by(|a, b| b.cmp(a)),
        }
        let rank = tokens.len() - end - 1;
        Value {
            state: word.get(rank).copied().unwrap_or(STATES - 2),
            positions,
            buffer_symbols,
        }
    }

    /// Independent rank counting with multiplicities; no sorting or producer
    /// execution. Ordinal comparisons are declared structure, not inferred
    /// invariance under arbitrary permutations of symbol identities.
    pub fn interpret(&self, tokens: &[usize]) -> (usize, usize) {
        let locations: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.delimiter).then_some(i))
            .collect();
        let endpoint = if self.latest {
            locations.last()
        } else {
            locations.first()
        };
        let Some(&end) = endpoint.filter(|&&i| i >= self.skip) else {
            return (STATES - 1, 0);
        };
        let body: Vec<_> = tokens
            .iter()
            .skip(self.skip)
            .take(end - self.skip)
            .copied()
            .collect();
        let rank = tokens.iter().skip(end + 1).count();
        let chosen = body.iter().enumerate().find(|&(i, &token)| {
            let position = match self.order {
                Order::Index => i,
                Order::ReverseIndex => body.len() - i - 1,
                Order::AscendingRank => body
                    .iter()
                    .enumerate()
                    .filter(|&(j, &v)| v < token || (v == token && j < i))
                    .count(),
                Order::DescendingRank => body
                    .iter()
                    .enumerate()
                    .filter(|&(j, &v)| v > token || (v == token && j < i))
                    .count(),
            };
            position == rank
        });
        (chosen.map_or(STATES - 2, |(_, &t)| t), body.len())
    }
}

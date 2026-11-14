//! Generic partial directed matchings of observed INPUT symbols. No labels,
//! task names, bracket pairs or target decoder enter catalogue construction.
use crate::*;
use std::collections::BTreeSet;
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Program {
    pub pairs: Vec<(usize, usize)>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    pub state: usize,
    pub positions: u64,
    pub peak_depth: usize,
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
    if !(2..=9).contains(&alphabet.len())
        || !(2..=1024).contains(&vocabulary)
        || alphabet.iter().any(|&t| t >= vocabulary)
        || alphabet.windows(2).any(|w| w[0] >= w[1])
    {
        return Err("nested catalogue requires2..9 distinct sorted observed symbols".into());
    }
    fn enumerate(rest: &[usize], pairs: &mut Vec<(usize, usize)>, out: &mut Vec<Program>) {
        if rest.len() < 2 || pairs.len() == 3 {
            if !pairs.is_empty() {
                out.push(Program {
                    pairs: pairs.clone(),
                });
            }
            return;
        }
        // The least remaining symbol is either unused or paired with a later
        // symbol. This visits every partial matching once, including both
        // directions of every pair, without permuting unnamed pair types.
        enumerate(&rest[1..], pairs, out);
        for i in 1..rest.len() {
            let tail: Vec<_> = rest
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != 0 && k != i)
                .map(|(_, &v)| v)
                .collect();
            for pair in [(rest[0], rest[i]), (rest[i], rest[0])] {
                pairs.push(pair);
                enumerate(&tail, pairs, out);
                pairs.pop();
            }
        }
    }
    let mut out = vec![];
    enumerate(alphabet, &mut vec![], &mut out);
    if out.len() > 16384 {
        return Err("nested candidate cap16384".into());
    }
    Ok(out)
}
impl Program {
    pub fn run(&self, tokens: &[usize]) -> Value {
        let (mut stack, mut invalid, mut peak) = (Vec::with_capacity(tokens.len()), false, 0);
        for &token in tokens {
            if let Some(&(_, expected)) = self.pairs.iter().find(|&&(push, _)| push == token) {
                stack.push(expected);
                peak = peak.max(stack.len());
            } else if self.pairs.iter().any(|&(_, pop)| pop == token) {
                invalid |= stack.pop() != Some(token);
            }
        }
        Value {
            state: if invalid {
                2
            } else {
                usize::from(!stack.is_empty())
            },
            positions: tokens.len() as u64,
            peak_depth: peak,
        }
    }
    /// Independent adjacent cancellation, plus an independent reflected
    /// count for peak storage. No producer stack or dictionary is called.
    pub fn interpret(&self, tokens: &[usize]) -> (usize, usize) {
        let mut word: Vec<_> = tokens
            .iter()
            .copied()
            .filter(|&t| self.pairs.iter().any(|&(a, b)| t == a || t == b))
            .collect();
        while let Some(i) = word
            .windows(2)
            .position(|w| self.pairs.contains(&(w[0], w[1])))
        {
            word.drain(i..i + 2);
        }
        let invalid = word
            .iter()
            .any(|&t| self.pairs.iter().any(|&(_, b)| t == b));
        let (mut level, mut peak) = (0usize, 0);
        for t in tokens {
            if self.pairs.iter().any(|(a, _)| a == t) {
                level += 1;
            } else if self.pairs.iter().any(|(_, b)| b == t) {
                level = level.saturating_sub(1);
            }
            peak = peak.max(level);
        }
        (
            if invalid {
                2
            } else {
                usize::from(!word.is_empty())
            },
            peak,
        )
    }
}

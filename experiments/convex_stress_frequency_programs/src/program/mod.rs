//! Finite input-only group, order and aggregate programs. Execution returns
//! latent quantities/structural states or an explicitly declared input copy.
//! It has no task name, target sequence, vocabulary answer map or decoder.
use crate::*;
mod bank;
mod independent;
#[cfg(test)]
mod tests;
pub use bank::{Roles, catalogue, infer};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Basis {
    Positions,
    Distinct,
    Maximum,
    Minimum,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Order {
    Source,
    Ascending,
    Descending,
    Reverse,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Statistic {
    Frequency,
    EqualFrequencyClasses,
    PrefixFrequency,
    Position,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Emit {
    Quantity,
    Copy,
    QuantityCopy,
    CopyQuantity,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Tail {
    None,
    Maximum,
    Minimum,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Program {
    pub vocabulary: usize,
    pub delimiter: usize,
    pub latest: bool,
    pub skip: usize,
    pub basis: Basis,
    pub order: Order,
    pub statistic: Statistic,
    pub emit: Emit,
    pub prefix: bool,
    pub padding: bool,
    pub tail: Tail,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Atom {
    Role(usize),
    Copy(usize),
    Quantity(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Value {
    pub atom: Atom,
    pub scalar_work: u64,
    pub buffer_symbols: usize,
}

impl Program {
    pub fn validate(&self) -> Result<()> {
        if !(2..=1024).contains(&self.vocabulary)
            || self.delimiter >= self.vocabulary
            || self.skip > 1
            || (self.emit == Emit::Copy && self.statistic != Statistic::Frequency)
            || (matches!(self.basis, Basis::Maximum | Basis::Minimum)
                && self.order != Order::Source)
        {
            return Err("invalid canonical aggregate program".into());
        }
        Ok(())
    }

    pub fn run(&self, tokens: &[usize]) -> Value {
        assert!(tokens.len() <= 4096 && tokens.iter().all(|&t| t < self.vocabulary));
        let end = if self.latest {
            tokens.iter().rposition(|&t| t == self.delimiter)
        } else {
            tokens.iter().position(|&t| t == self.delimiter)
        };
        let Some(end) = end.filter(|&end| end > self.skip) else {
            return Value {
                atom: Atom::Role(2),
                scalar_work: 16 * tokens.len() as u64 + 16,
                buffer_symbols: 0,
            };
        };
        let word = &tokens[self.skip..end];
        let mut counts = vec![0usize; self.vocabulary];
        let mut first = vec![usize::MAX; self.vocabulary];
        let mut prefix_counts = Vec::with_capacity(word.len());
        let mut distinct = vec![];
        for (i, &t) in word.iter().enumerate() {
            if counts[t] == 0 {
                first[t] = i;
                distinct.push(t);
            }
            counts[t] += 1;
            prefix_counts.push(counts[t]);
        }
        let maximum = *distinct
            .iter()
            .max_by_key(|&&t| (counts[t], usize::MAX - first[t]))
            .unwrap();
        let minimum = *distinct
            .iter()
            .min_by_key(|&&t| (counts[t], first[t]))
            .unwrap();
        let mut indices: Vec<_> = match self.basis {
            Basis::Positions => (0..word.len()).collect(),
            Basis::Distinct => distinct.iter().map(|&t| first[t]).collect(),
            Basis::Maximum => vec![first[maximum]],
            Basis::Minimum => vec![first[minimum]],
        };
        match self.order {
            Order::Source => {}
            Order::Ascending => indices.sort_by_key(|&i| (counts[word[i]], i)),
            Order::Descending => indices.sort_by_key(|&i| (std::cmp::Reverse(counts[word[i]]), i)),
            Order::Reverse => indices.reverse(),
        }
        let width = if matches!(self.emit, Emit::QuantityCopy | Emit::CopyQuantity) {
            2
        } else {
            1
        };
        let body = indices.len() * width;
        let padded = if self.padding {
            body.max(word.len())
        } else {
            body
        };
        let rank = tokens.len() - end - 1;
        let atom = if self.prefix && rank == 0 {
            Atom::Role(0)
        } else {
            let at = rank - usize::from(self.prefix);
            if at < body {
                let i = indices[at / width];
                let copy = self.emit == Emit::Copy
                    || (self.emit == Emit::CopyQuantity && at.is_multiple_of(width))
                    || (self.emit == Emit::QuantityCopy && at % width == 1);
                if copy {
                    Atom::Copy(word[i])
                } else {
                    let q = match self.statistic {
                        Statistic::Frequency => counts[word[i]],
                        Statistic::EqualFrequencyClasses => distinct
                            .iter()
                            .filter(|&&t| counts[t] == counts[word[i]])
                            .count(),
                        Statistic::PrefixFrequency => prefix_counts[i],
                        Statistic::Position => i + 1,
                    };
                    Atom::Quantity(q)
                }
            } else if at < padded {
                Atom::Role(0)
            } else if at == padded && self.tail != Tail::None {
                Atom::Copy(if self.tail == Tail::Maximum {
                    maximum
                } else {
                    minimum
                })
            } else {
                Atom::Role(1)
            }
        };
        // Includes count tables, comparisons, sorting, all candidate aggregate
        // scans and bounds checks; independent rendering is charged separately.
        Value {
            atom,
            scalar_work: (64 * (tokens.len() + self.vocabulary + word.len() * word.len()) + 128)
                as u64,
            buffer_symbols: 2 * self.vocabulary + 4 * word.len(),
        }
    }
}

//! Independent specification interpreter: recount each source token and
//! render the entire latent sequence, then index it. No producer count table,
//! position selection or rank arithmetic is reused.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

impl Program {
    pub fn interpret(&self, tokens: &[usize]) -> Atom {
        let occurrences: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.delimiter).then_some(i))
            .collect();
        let chosen = if self.latest {
            occurrences.last()
        } else {
            occurrences.first()
        };
        let Some(&end) = chosen.filter(|&&end| end > self.skip) else {
            return Atom::Role(2);
        };
        let word: Vec<_> = tokens.iter().take(end).skip(self.skip).copied().collect();
        let distinct: BTreeSet<_> = word.iter().copied().collect();
        let recount: BTreeMap<_, _> = distinct
            .iter()
            .map(|&t| (t, word.iter().filter(|&&x| x == t).count()))
            .collect();
        let count = |t| recount[&t];
        let maximum = (0..word.len())
            .min_by_key(|&i| (std::cmp::Reverse(count(word[i])), i))
            .unwrap();
        let minimum = (0..word.len())
            .min_by_key(|&i| (count(word[i]), i))
            .unwrap();
        let mut chosen: Vec<_> = (0..word.len())
            .filter(|&i| match self.basis {
                Basis::Positions => true,
                Basis::Distinct => !word[..i].contains(&word[i]),
                Basis::Maximum => i == maximum,
                Basis::Minimum => i == minimum,
            })
            .collect();
        chosen.sort_by(|&i, &j| match self.order {
            Order::Source => i.cmp(&j),
            Order::Reverse => j.cmp(&i),
            Order::Ascending => (count(word[i]), i).cmp(&(count(word[j]), j)),
            Order::Descending => {
                (std::cmp::Reverse(count(word[i])), i).cmp(&(std::cmp::Reverse(count(word[j])), j))
            }
        });
        let mut body = vec![];
        for i in chosen {
            let q = match self.statistic {
                Statistic::Frequency => count(word[i]),
                Statistic::EqualFrequencyClasses => distinct
                    .iter()
                    .filter(|&&t| count(t) == count(word[i]))
                    .count(),
                Statistic::PrefixFrequency => word[..=i].iter().filter(|&&t| t == word[i]).count(),
                Statistic::Position => i + 1,
            };
            let pair = match self.emit {
                Emit::Quantity => vec![Atom::Quantity(q)],
                Emit::Copy => vec![Atom::Copy(word[i])],
                Emit::QuantityCopy => vec![Atom::Quantity(q), Atom::Copy(word[i])],
                Emit::CopyQuantity => vec![Atom::Copy(word[i]), Atom::Quantity(q)],
            };
            body.extend(pair);
        }
        if self.padding {
            body.resize(body.len().max(word.len()), Atom::Role(0));
        }
        match self.tail {
            Tail::None => {}
            Tail::Maximum => body.push(Atom::Copy(word[maximum])),
            Tail::Minimum => body.push(Atom::Copy(word[minimum])),
        }
        if self.prefix {
            body.insert(0, Atom::Role(0));
        }
        body.push(Atom::Role(1));
        body.get(tokens.iter().skip(end + 1).count())
            .copied()
            .unwrap_or(Atom::Role(1))
    }
}

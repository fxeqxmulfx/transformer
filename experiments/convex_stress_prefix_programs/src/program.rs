//! Input-only finite count, position and adjacent-copy summaries. Raw output
//! labels are absent from execution; structural states have learned decoders.
use crate::*;
mod roles;
pub use roles::{Roles, catalogue, infer};
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Emit {
    Final,
    All,
    Selected,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Side {
    Before,
    After,
}
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Program {
    pub delimiter: usize,
    pub counter: usize,
    pub modulus: usize,
    pub skip: usize,
    pub stride: usize,
    pub offset: usize,
    pub latest: bool,
    pub emit: Emit,
    pub copy: Option<Side>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Value {
    pub state: usize,
    pub positions: u64,
    pub buffer_symbols: usize,
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
                state: 255,
                positions: tokens.len() as u64,
                buffer_symbols: 0,
            };
        };
        let indices: Vec<_> = (self.skip + self.offset..end)
            .step_by(self.stride)
            .collect();
        let mut count = 0;
        let mut events = vec![];
        for &i in &indices {
            count += usize::from(tokens[i] == self.counter);
            if self.emit != Emit::Selected || tokens[i] == self.counter {
                events.push((i, count));
            }
        }
        let rank = tokens.len() - end - 1;
        let state = if self.emit == Emit::Final {
            if rank == 0 {
                240 + count % self.modulus
            } else {
                254
            }
        } else if rank == 0 {
            240
        } else {
            let width = if self.copy.is_some() { 2 } else { 1 };
            match events.get((rank - 1) / width) {
                None => 254,
                Some(&(i, count)) => {
                    if let Some(side) = self.copy.filter(|_| (rank - 1).is_multiple_of(width)) {
                        let at = match side {
                            Side::Before => i.checked_sub(1),
                            Side::After => i.checked_add(1),
                        };
                        at.filter(|&p| p >= self.skip && p < end)
                            .map_or(255, |p| tokens[p])
                    } else {
                        240 + count % self.modulus
                    }
                }
            }
        };
        Value {
            state,
            positions: (tokens.len() + 2 * indices.len() + events.len()) as u64,
            buffer_symbols: indices.len() + 2 * events.len(),
        }
    }

    /// Independent positional selection and prefix recounting. No producer
    /// execution or running-count table is used by this interpreter.
    pub fn interpret(&self, tokens: &[usize]) -> (usize, usize) {
        let ends: Vec<_> = tokens
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| (t == self.delimiter).then_some(i))
            .collect();
        let endpoint = if self.latest {
            ends.last()
        } else {
            ends.first()
        };
        let Some(&end) = endpoint.filter(|&&i| i >= self.skip) else {
            return (255, 0);
        };
        let positions: Vec<_> = (self.skip..end)
            .filter(|&i| (i - self.skip) % self.stride == self.offset)
            .collect();
        let emitted: Vec<_> = positions
            .iter()
            .copied()
            .filter(|&i| self.emit != Emit::Selected || tokens[i] == self.counter)
            .collect();
        let buffer = positions.len() + 2 * emitted.len();
        let rank = tokens.iter().skip(end + 1).count();
        let count = |last: usize| {
            positions
                .iter()
                .filter(|&&i| i <= last && tokens[i] == self.counter)
                .count()
                % self.modulus
        };
        if self.emit == Emit::Final {
            return (if rank == 0 { 240 + count(end) } else { 254 }, buffer);
        }
        if rank == 0 {
            return (240, buffer);
        }
        let event = if self.copy.is_some() {
            (rank - 1) / 2
        } else {
            rank - 1
        };
        let Some(&i) = emitted.get(event) else {
            return (254, buffer);
        };
        if self.copy.is_none() || rank.is_multiple_of(2) {
            return (240 + count(i), buffer);
        }
        let selected = (self.skip..end).find(|&p| match self.copy.unwrap() {
            Side::Before => p + 1 == i,
            Side::After => i + 1 == p,
        });
        (selected.map_or(255, |p| tokens[p]), buffer)
    }
}

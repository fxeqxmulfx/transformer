//! Frozen dense state evaluator from a440f6a0. Kept only for numerical
//! equivalence checks of the sparse storage implementation; never a trainer.
use crate::{C, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Transport {
    Residual,
    Fourier,
}

impl Transport {
    /// CD3's exact D F4 D* formula and (1+i*a)/sqrt(1+a*a) phase convention.
    pub fn apply(self, layer: usize, input: [C; 4]) -> [C; 4] {
        if self == Self::Residual {
            return input;
        }
        let angles = [
            0.,
            0.25 * (layer + 1) as f64,
            -0.5 * (layer + 1) as f64,
            0.75 * (layer + 1) as f64,
        ];
        let phases = angles.map(|a| C::new(1., a) / 1_f64.hypot(a));
        let roots = [
            C::new(1., 0.),
            C::new(0., 1.),
            C::new(-1., 0.),
            C::new(0., -1.),
        ];
        std::array::from_fn(|row| {
            (0..4)
                .map(|col| {
                    phases[row] * roots[(row * col) % 4] * phases[col].conj() * input[col] / 2.
                })
                .sum()
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Mask {
    All,
    Pair(usize, usize),
}
impl Mask {
    fn contains(&self, token: usize) -> bool {
        match *self {
            Self::All => true,
            Self::Pair(a, b) => token == a || token == b,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Block {
    PreviousDistinct,
    CompletedSuccessor,
}

#[derive(Clone)]
struct State {
    amplitudes: Vec<[C; 4]>,
    history: Vec<usize>,
}
impl State {
    fn norm(&self) -> f64 {
        self.amplitudes
            .iter()
            .flatten()
            .map(C::norm_sqr)
            .sum::<f64>()
            .sqrt()
    }
    fn position(&self) -> usize {
        let mut best = 0;
        let energy = |row: &[C; 4]| row.iter().map(C::norm_sqr).sum::<f64>();
        for i in 1..self.amplitudes.len() {
            if energy(&self.amplitudes[i])
                .total_cmp(&energy(&self.amplitudes[best]))
                .is_gt()
            {
                best = i;
            }
        }
        best
    }
    fn token(&self, prefix: &[i64], vocab: usize) -> usize {
        prefix.get(self.position()).map_or(vocab, |t| *t as usize)
    }
    fn initial(prefix: &[i64], mask: &Mask, vocab: usize) -> Self {
        let position = prefix
            .iter()
            .rposition(|t| mask.contains(*t as usize))
            .unwrap_or(prefix.len());
        let mut amplitudes = vec![[C::default(); 4]; prefix.len() + 1];
        amplitudes[position][0] = C::new(1., 0.);
        Self {
            amplitudes,
            history: vec![prefix.get(position).map_or(vocab, |t| *t as usize)],
        }
    }
    fn step(
        &self,
        prefix: &[i64],
        mask: &Mask,
        vocab: usize,
        block: Block,
        transport: Transport,
        layer: usize,
    ) -> Result<Self> {
        let current = self.position();
        let target = if current == prefix.len() {
            current
        } else {
            let token = prefix[current];
            match block {
                Block::PreviousDistinct => (0..current)
                    .rev()
                    .find(|&i| prefix[i] != token && mask.contains(prefix[i] as usize))
                    .unwrap_or(prefix.len()),
                Block::CompletedSuccessor => {
                    let positions: Vec<_> = prefix
                        .iter()
                        .enumerate()
                        .filter(|(_, t)| mask.contains(**t as usize))
                        .map(|(i, _)| i)
                        .collect();
                    positions
                        .windows(2)
                        .rev()
                        .find(|pair| prefix[pair[0]] == token)
                        .map_or(prefix.len(), |pair| pair[1])
                }
            }
        };
        let divisor = self.norm().max(1.);
        let mut capped = self.amplitudes.clone();
        for value in capped.iter_mut().flatten() {
            *value /= divisor;
        }
        let mut moved = vec![[C::default(); 4]; capped.len()];
        moved[target] = capped[current];
        let mut amplitudes = Vec::with_capacity(capped.len());
        let mut branch_norm_squared = 0.;
        for (i, old) in self.amplitudes.iter().enumerate() {
            let delta = std::array::from_fn(|stream| moved[i][stream] - capped[i][stream]);
            let branch = transport.apply(layer, delta);
            branch_norm_squared += branch.iter().map(C::norm_sqr).sum::<f64>();
            let skip = transport.apply(layer, *old);
            amplitudes.push(std::array::from_fn(|stream| skip[stream] + branch[stream]));
        }
        let mut result = Self {
            amplitudes,
            history: self.history.clone(),
        };
        if branch_norm_squared > 4. + 1e-10
            || result.norm() > self.norm() + 2. + 1e-10
            || !result.norm().is_finite()
            || (result.norm() - 1.).abs() > 1e-10
        {
            return Err("complex path norm certificate failed".into());
        }
        result.history.push(result.token(prefix, vocab));
        Ok(result)
    }
}

pub(super) fn encode(bank: &super::Bank, prefix: &[i64]) -> Result<Vec<usize>> {
    let transport = match bank.transport {
        super::Transport::Residual => Transport::Residual,
        super::Transport::Fourier => Transport::Fourier,
    };
    let mut features = vec![];
    for (m, mask) in bank.masks.iter().enumerate() {
        let mask = match *mask {
            super::Mask::All => Mask::All,
            super::Mask::Pair(a, b) => Mask::Pair(a, b),
        };
        let mut states = vec![State::initial(prefix, &mask, bank.vocab)];
        for layer in 0..bank.depth {
            let mut next = vec![];
            for state in states {
                for block in [Block::PreviousDistinct, Block::CompletedSuccessor] {
                    next.push(state.step(prefix, &mask, bank.vocab, block, transport, layer)?);
                }
            }
            states = next;
        }
        for (p, state) in states.into_iter().enumerate() {
            let base = (m * (1 << bank.depth) + p) * bank.stride();
            let mut hash = 0xcbf29ce484222325u64;
            for token in state.history {
                for byte in (token as u64).to_le_bytes() {
                    hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                }
            }
            let selected = state
                .amplitudes
                .iter()
                .enumerate()
                .max_by(|a, b| {
                    let energy = |r: &[C; 4]| r.iter().map(C::norm_sqr).sum::<f64>();
                    energy(a.1).total_cmp(&energy(b.1)).then(b.0.cmp(&a.0))
                })
                .unwrap()
                .0;
            features.push(base + prefix.get(selected).map_or(bank.vocab, |t| *t as usize));
            features.push(base + bank.vocab + 1 + hash as usize % bank.history_slots);
        }
    }
    Ok(features)
}

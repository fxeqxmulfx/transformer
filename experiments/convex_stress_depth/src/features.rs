//! Generic causal path library. No target, task name or arithmetic rule enters
//! a block. Four complex streams encode a position in the visible prefix.
//! Each branch reads the preceding path's actual amplitudes, moves their selected
//! content, and writes a bounded residual through a fixed transport.
//! Residual/Fourier equality is a coordinate-invariance control: routing reads
//! stream energy, and the whole branch displacement is transported as well.
//! It is not a claim that Fourier mixing improves accuracy over identity.
use crate::{C, Result, data};
use serde::{Deserialize, Serialize};

pub const MAX_PREFIX: usize = 1024;
pub const MAX_DEPTH: usize = 4;
pub const MAX_VOCAB: usize = 1024;

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Bank {
    pub vocab: usize,
    pub depth: usize,
    pub history_slots: usize,
    pub masks: Vec<Mask>,
    pub transport: Transport,
}

impl Bank {
    /// Masks use only frequencies in original training input tokens. Include
    /// every pair of the four most frequent symbols, plus the unfiltered view.
    pub fn discover(
        vocab: usize,
        depth: usize,
        history_slots: usize,
        transport: Transport,
        train: &[data::Example],
    ) -> Result<Self> {
        if !(2..=MAX_VOCAB).contains(&vocab) || train.is_empty() {
            return Err("invalid vocabulary or empty training inputs".into());
        }
        let mut counts = vec![0u64; vocab];
        for example in train {
            for &token in &example.tokens {
                if token < 0 || token as usize >= vocab {
                    return Err("invalid training input token".into());
                }
                counts[token as usize] += 1;
            }
        }
        let mut ranked: Vec<_> = (0..vocab).filter(|&t| counts[t] > 0).collect();
        ranked.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
        ranked.truncate(4);
        ranked.sort_unstable();
        let mut masks = vec![Mask::All];
        for (i, &a) in ranked.iter().enumerate() {
            for &b in &ranked[i + 1..] {
                masks.push(Mask::Pair(a, b));
            }
        }
        let bank = Self {
            vocab,
            depth,
            history_slots,
            masks,
            transport,
        };
        bank.validate()?;
        Ok(bank)
    }
    pub fn validate(&self) -> Result<()> {
        if !(2..=MAX_VOCAB).contains(&self.vocab)
            || self.depth > MAX_DEPTH
            || self.history_slots == 0
            || self.history_slots > 4096
            || self.masks.is_empty()
            || self.masks.len() > 7
            || self.masks[0] != Mask::All
            || self
                .masks
                .iter()
                .any(|m| matches!(*m,Mask::Pair(a,b) if a>=b||b>=self.vocab))
        {
            return Err("invalid fixed path bank".into());
        }
        Ok(())
    }
    pub fn paths(&self) -> usize {
        self.masks.len() * (1 << self.depth)
    }
    pub fn stride(&self) -> usize {
        self.vocab + 1 + self.history_slots
    }
    pub fn dimension(&self) -> usize {
        self.paths() * self.stride()
    }
    /// Exactly two unit features per path: final visible symbol and a fixed
    /// bounded hash of the executed history. Hash collisions are retained.
    pub fn encode(&self, prefix: &[i64]) -> Result<Vec<usize>> {
        self.validate()?;
        if prefix.is_empty()
            || prefix.len() > MAX_PREFIX
            || prefix.iter().any(|t| *t < 0 || *t as usize >= self.vocab)
        {
            return Err("invalid causal prefix".into());
        }
        let mut features = Vec::with_capacity(2 * self.paths());
        for (mask_index, mask) in self.masks.iter().enumerate() {
            let mut states = vec![State::initial(prefix, mask, self.vocab)];
            for layer in 0..self.depth {
                let mut next = Vec::with_capacity(2 * states.len());
                for state in states {
                    for block in [Block::PreviousDistinct, Block::CompletedSuccessor] {
                        next.push(state.step(
                            prefix,
                            mask,
                            self.vocab,
                            block,
                            self.transport,
                            layer,
                        )?);
                    }
                }
                states = next;
            }
            for (path, state) in states.into_iter().enumerate() {
                let index = mask_index * (1 << self.depth) + path;
                let mut hash = 0xcbf29ce484222325u64;
                for token in &state.history {
                    for byte in (*token as u64).to_le_bytes() {
                        hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
                    }
                }
                features.push(index * self.stride() + state.token(prefix, self.vocab));
                features.push(
                    index * self.stride() + self.vocab + 1 + hash as usize % self.history_slots,
                );
            }
        }
        Ok(features)
    }
    /// Explicit coefficient, gradient, projection scratch, feature-cache and
    /// observation/index buffers. This is
    /// an allocation estimate; a separate OS limit must cover total process RSS.
    pub fn allocation_bytes(&self, observations: usize) -> Result<usize> {
        self.validate()?;
        self.dimension()
            .checked_mul(self.vocab)
            .and_then(|n| n.checked_mul(16))
            .and_then(|n| {
                self.dimension()
                    .checked_mul(8)
                    .and_then(|m| n.checked_add(m))
            })
            .and_then(|n| {
                self.vocab
                    .checked_mul(64)
                    .and_then(|m| n.checked_add(m + 8))
            })
            .and_then(|n| {
                observations
                    .checked_mul(self.paths())
                    .and_then(|m| m.checked_mul(8))
                    .and_then(|m| observations.checked_mul(32).and_then(|o| m.checked_add(o)))
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or_else(|| "path-bank allocation overflow".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_fourier_transport_preserves_arbitrary_complex_norms() {
        for layer in 0..8 {
            let input = [
                C::new(1., 2.),
                C::new(-3., 0.4),
                C::new(0.5, -0.8),
                C::new(2., -1.),
            ];
            let output = Transport::Fourier.apply(layer, input);
            assert!(
                (output.iter().map(C::norm_sqr).sum::<f64>()
                    - input.iter().map(C::norm_sqr).sum::<f64>())
                .abs()
                    < 1e-12
            );
        }
    }
    #[test]
    fn deeper_paths_follow_their_own_previous_state() -> Result<()> {
        let prefix = [9, 7, 7, 8, 3, 3, 8, 7];
        let mask = Mask::Pair(3, 7);
        let mut state = State::initial(&prefix, &mask, 12);
        for layer in 0..3 {
            state = state.step(
                &prefix,
                &mask,
                12,
                Block::PreviousDistinct,
                Transport::Fourier,
                layer,
            )?;
        }
        assert_eq!(state.history, vec![7, 3, 7, 12]);
        assert!((state.norm() - 1.).abs() < 1e-12);
        Ok(())
    }
    #[test]
    fn completed_successors_exclude_the_unfinished_current_occurrence() -> Result<()> {
        let prefix = [2, 8, 4, 9, 2];
        let state = State::initial(&prefix, &Mask::All, 10).step(
            &prefix,
            &Mask::All,
            10,
            Block::CompletedSuccessor,
            Transport::Residual,
            0,
        )?;
        assert_eq!(state.token(&prefix, 10), 8);
        Ok(())
    }
    #[test]
    fn energy_readout_is_invariant_under_the_matched_coordinate_transport() -> Result<()> {
        for depth in 0..=MAX_DEPTH {
            let mut bank = Bank {
                vocab: 12,
                depth,
                history_slots: 4096,
                masks: vec![Mask::All, Mask::Pair(3, 7)],
                transport: Transport::Residual,
            };
            let prefix = [9, 7, 7, 8, 3, 3, 8, 7];
            let plain = bank.encode(&prefix)?;
            bank.transport = Transport::Fourier;
            assert_eq!(plain, bank.encode(&prefix)?);
            assert_eq!(plain.len(), 2 * bank.paths());
            assert!(plain.windows(2).all(|p| p[0] < p[1]));
            assert!(plain.iter().all(|i| *i < bank.dimension()));
        }
        Ok(())
    }
    #[test]
    fn depth_prefix_and_allocation_limits_are_checked_before_work() -> Result<()> {
        let mut bank = Bank {
            vocab: 548,
            depth: 4,
            history_slots: 4096,
            masks: vec![Mask::All],
            transport: Transport::Fourier,
        };
        assert!(bank.allocation_bytes(usize::MAX).is_err());
        assert!(bank.encode(&vec![1; MAX_PREFIX + 1]).is_err());
        bank.depth = MAX_DEPTH + 1;
        assert!(bank.encode(&[1]).is_err());
        Ok(())
    }

    #[test]
    fn discovery_reads_input_frequencies_without_labels_or_task_names() -> Result<()> {
        let example = data::Example {
            task: "first".into(),
            tokens: vec![2, 2, 3, 2, 5, 3],
            targets: vec![0; 6],
            changes: vec![false; 6],
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        };
        let first = Bank::discover(
            8,
            2,
            128,
            Transport::Fourier,
            std::slice::from_ref(&example),
        )?;
        let mut changed = example;
        changed.task = "different".into();
        changed.targets = vec![7; 6];
        let second = Bank::discover(
            8,
            2,
            128,
            Transport::Fourier,
            std::slice::from_ref(&changed),
        )?;
        assert_eq!(first.masks, second.masks);
        assert_eq!(
            first.masks,
            vec![
                Mask::All,
                Mask::Pair(2, 3),
                Mask::Pair(2, 5),
                Mask::Pair(3, 5)
            ]
        );
        assert_eq!(
            first.encode(&changed.tokens)?,
            second.encode(&changed.tokens)?
        );
        assert!(Bank::discover(usize::MAX, 2, 128, Transport::Fourier, &[changed]).is_err());
        Ok(())
    }
}

//! FixedEvaluation.cached_scatter justifies checking the complete role code
//! while keeping each floating-point addition at its original position.
use super::{Agreement, Witness};

const SLOTS: usize = 1024;

pub(super) struct Coefficients<'a> {
    table: &'a [f32],
    agreement: &'a Agreement,
    slots: Vec<Option<(u32, f32)>>,
}

impl<'a> Coefficients<'a> {
    pub(super) fn new(table: &'a [f32], agreement: &'a Agreement) -> Self {
        Self {
            table,
            agreement,
            slots: vec![None; SLOTS],
        }
    }

    pub(super) fn vote(&mut self, path: Witness, votes: &mut [f32]) {
        let cell = &mut self.slots[path.code as usize % SLOTS];
        let weight = match *cell {
            Some((tag, value)) if tag == path.code => value,
            _ => {
                let value = self.agreement.weight(self.table, path);
                *cell = Some((path.code, value));
                value
            }
        };
        votes[path.position] += weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed_lookup::BINS;

    #[test]
    fn collisions_position_changes_and_new_weights_preserve_each_rounded_sum() {
        let low = 0x8020_0009;
        let high = low ^ 0x4740_8400;
        assert_eq!(low as usize % SLOTS, high as usize % SLOTS);
        for scale in [1., -0.4] {
            let table: Vec<_> = (0..BINS).map(|i| scale * ((i % 19) as f32 - 9.)).collect();
            let rule = Agreement::new(&table, 4);
            assert_ne!(
                rule.weight(&table, Witness::at_code(low)),
                rule.weight(&table, Witness::at_code(high))
            );
            let mut direct = [0f32; 3];
            let mut actual = direct;
            let mut cached = Coefficients::new(&table, &rule);
            for (position, code) in [
                (1, low),
                (2, low),
                (1, high),
                (1, low),
                (0, high),
                (2, high),
            ] {
                let path = Witness { position, code };
                direct[position] += rule.weight(&table, path);
                cached.vote(path, &mut actual);
                assert_eq!(actual.map(f32::to_bits), direct.map(f32::to_bits));
            }
        }
    }
}

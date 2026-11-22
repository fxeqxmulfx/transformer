//! Sets of up to 256 learned image slots in the correspondence searches.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Values([u64; 4]);

impl Values {
    pub const ALL: Self = Self([u64::MAX; 4]);

    pub fn insert(&mut self, slot: usize) {
        self.0[slot / 64] |= 1 << (slot % 64);
    }

    pub fn intersect(&mut self, other: Self) {
        for (word, mask) in self.0.iter_mut().zip(other.0) {
            *word &= mask;
        }
    }

    pub fn is_empty(self) -> bool {
        self.0 == [0; 4]
    }

    pub fn len(self) -> u32 {
        self.0.iter().map(|word| word.count_ones()).sum()
    }

    pub fn members(self) -> impl Iterator<Item = usize> {
        self.0
            .into_iter()
            .enumerate()
            .flat_map(|(block, mut bits)| {
                std::iter::from_fn(move || {
                    if bits == 0 {
                        return None;
                    }
                    let slot = block * 64 + bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    Some(slot)
                })
            })
    }

    pub fn unique(self) -> Option<u32> {
        if self.len() != 1 {
            return None;
        }
        self.members().next().map(|i| i as u32)
    }
}

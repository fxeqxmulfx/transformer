//! CM72 construction_shared_eq, proved first at Lean 1279ab89. Share only
//! when every unknown row in this block stores one identical binary64 word.
//! Arbitrary nonuniform rows keep the full original construction loop.
use super::*;

pub(super) struct Construction {
    classes: [usize; 5],
    count: usize,
    representative: usize,
}
impl Construction {
    pub fn new(domain: &Domain, q: &Parameters, row: usize, support: &Support) -> Option<Self> {
        if !support.all_classes() || support.known().len() + 1 >= domain.shape.classes {
            return None;
        }
        for r in row..row + 4 {
            if domain.labels[r].is_none() {
                let unknown = (domain.offsets[r] - domain.shape.programs) / domain.shape.classes;
                q.unknown_rows().repeated(unknown)?;
            }
        }
        let representative = (0..domain.shape.classes).find(|c| !support.known().contains(c))?;
        let mut classes = [0; 5];
        classes[..support.known().len()].copy_from_slice(support.known());
        classes[support.known().len()] = representative;
        let count = support.known().len() + 1;
        classes[..count].sort_unstable();
        Some(Self {
            classes,
            count,
            representative,
        })
    }
    pub fn classes(&self) -> &[usize] {
        &self.classes[..self.count]
    }
    pub fn expand(&self, values: &mut [storage::Column]) {
        let column = values[self.representative];
        for (c, value) in values.iter_mut().enumerate() {
            if !self.classes().contains(&c) {
                *value = column;
            }
        }
    }
}

#[cfg(test)]
mod tests;

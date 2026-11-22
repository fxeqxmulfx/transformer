//! Unique available images for a fixed row of the triple graph.
//! The unowned part of each anchor pair is independent of the current query.
use super::ABSENT;
use std::collections::HashMap;

#[derive(Clone, Copy)]
enum Matches {
    Empty,
    One(u32),
    Multiple,
}

impl Matches {
    fn including(self, image: u32) -> Self {
        match self {
            Self::Empty => Self::One(image),
            _ => Self::Multiple,
        }
    }
}

pub(super) struct RowCandidates {
    free: Vec<usize>,
    buckets: HashMap<usize, HashMap<u32, Matches>>,
}

impl RowCandidates {
    pub(super) fn new(slots: usize, points: &[(usize, usize)]) -> Self {
        let mut occupied = vec![false; slots];
        for &(_, image) in points {
            occupied[image] = true;
        }
        Self {
            free: (0..slots).filter(|&image| !occupied[image]).collect(),
            buckets: HashMap::new(),
        }
    }

    /// The offset identifies the two output anchors. Every anchor is occupied.
    /// The direct image of the query, when admissible, is disjoint from the bucket.
    pub(super) fn unique(
        &mut self,
        offset: usize,
        classes: &[u32],
        key: u32,
        owned: u32,
        anchors: (usize, usize),
    ) -> Option<u32> {
        let mut matches = if self.free.len() <= 8 {
            let mut found = self.free.iter().copied().filter(|&y| classes[y] == key);
            match (found.next(), found.next()) {
                (None, _) => Matches::Empty,
                (Some(y), None) => Matches::One(y as u32),
                _ => Matches::Multiple,
            }
        } else {
            let index = self.buckets.entry(offset).or_insert_with(|| {
                let mut index = HashMap::with_capacity(self.free.len());
                for &y in &self.free {
                    index
                        .entry(classes[y])
                        .and_modify(|hits| *hits = Matches::Multiple)
                        .or_insert(Matches::One(y as u32));
                }
                index
            });
            index.get(&key).copied().unwrap_or(Matches::Empty)
        };
        if owned != ABSENT
            && owned != anchors.0 as u32
            && owned != anchors.1 as u32
            && classes[owned as usize] == key
        {
            matches = matches.including(owned);
        }
        match matches {
            Matches::One(y) => Some(y),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_index_agrees_with_the_full_candidate_scan() {
        let slots = 19;
        for occupied in 2..=slots {
            let points: Vec<_> = (0..occupied).map(|i| (i, i)).collect();
            let mut index = RowCandidates::new(slots, &points);
            for modulus in [1, 3, 7, 19] {
                let classes: Vec<_> = (0..slots).map(|y| ((y + 3) % modulus) as u32).collect();
                for owned in (0..occupied as u32).map(Some).chain([None]) {
                    for key in 0..=modulus as u32 {
                        let mut scan = (2..slots).filter(|&y| {
                            (y >= occupied || owned == Some(y as u32)) && classes[y] == key
                        });
                        let expected = scan
                            .next()
                            .filter(|_| scan.next().is_none())
                            .map(|y| y as u32);
                        assert_eq!(
                            index.unique(modulus, &classes, key, owned.unwrap_or(ABSENT), (0, 1)),
                            expected
                        );
                    }
                }
            }
        }
    }
}

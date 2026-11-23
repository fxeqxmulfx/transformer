//! Ground-truth correctness belongs to the original external scorer.
use crate::{data::Example, engine::Engine, metrics};
use minibatch::Result;
use serde_json::{Value, json};

pub struct RowScore {
    pub exact: usize,
    pub total: usize,
    outcomes: Vec<bool>,
}

impl RowScore {
    pub fn measure(engine: &mut Engine, examples: &[Example]) -> Result<Self> {
        let mut outcomes = Vec::with_capacity(examples.len());
        let mut ordered = true;
        let counts = metrics::evaluate_with_rows(engine, examples, true, |i, correct| {
            ordered &= i == outcomes.len();
            outcomes.push(correct);
        })?;
        let total = outcomes.len();
        let exact = outcomes.iter().filter(|&&b| b).count();
        if !ordered
            || total != examples.len()
            || counts["exact"] != exact
            || counts["examples"] != total
        {
            return Err("paired row outcomes differ from the complete GPT score".into());
        }
        Ok(Self {
            exact,
            total,
            outcomes,
        })
    }

    pub fn compare(&self, later: &Self) -> Result<Value> {
        if self.total != later.total {
            return Err("paired scores refer to different numbers of examples".into());
        }
        let mut both = 0;
        let mut gained = 0;
        let mut lost = 0;
        let mut neither = 0;
        for (&before, &after) in self.outcomes.iter().zip(&later.outcomes) {
            match (before, after) {
                (true, true) => both += 1,
                (false, true) => gained += 1,
                (true, false) => lost += 1,
                (false, false) => neither += 1,
            }
        }
        Ok(json!({"both_correct":both,"gained":gained,"lost":lost,"both_wrong":neither}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn equal_counts_can_hide_exchanged_answers() -> Result<()> {
        let first = RowScore {
            exact: 2,
            total: 4,
            outcomes: vec![true, true, false, false],
        };
        let later = RowScore {
            exact: 2,
            total: 4,
            outcomes: vec![true, false, true, false],
        };
        assert_eq!(
            first.compare(&later)?,
            json!({"both_correct":1,"gained":1,"lost":1,"both_wrong":1})
        );
        Ok(())
    }
}

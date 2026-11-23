//! Keep the source metric in the scorer and out of the model forward.
use super::*;

#[derive(Clone, Copy)]
pub(super) enum SplitScorer<'a> {
    Exact,
    Numeric(&'a numeric_probe::Teachers),
    TokenLoss,
}

impl<'a> SplitScorer<'a> {
    pub(super) fn new(case: &Value, teachers: Option<&'a numeric_probe::Teachers>) -> Self {
        if let Some(teachers) = teachers {
            Self::Numeric(teachers)
        } else if case["metric"] == "negative_loss" {
            Self::TokenLoss
        } else {
            Self::Exact
        }
    }

    pub(super) fn field(self) -> Option<&'static str> {
        match self {
            Self::Exact => None,
            Self::Numeric(_) => Some("mse"),
            Self::TokenLoss => Some("loss"),
        }
    }

    pub(super) fn scalar(self, observation: &Value) -> Result<Option<f64>> {
        let field = self.field();
        for key in ["mse", "loss"] {
            if observation.get(key).is_some() != (field == Some(key)) {
                return Err("observation metric differs from the source protocol".into());
            }
        }
        field
            .map(|key| {
                observation[key]
                    .as_f64()
                    .filter(|x| x.is_finite() && *x >= 0.)
                    .ok_or_else(|| "invalid scalar observation".into())
            })
            .transpose()
    }

    pub(super) fn score(self, engine: &mut Engine, rows: &[Example], split: &str) -> Result<Value> {
        if let Self::Numeric(teachers) = self {
            return teachers.score(engine, split, rows);
        }
        let score = metrics::evaluate(engine, rows, true)?;
        let correct = score["exact"]
            .as_u64()
            .ok_or("missing exact-sequence score")?;
        let mut counts = json!({"correct":correct,"total":rows.len()});
        if matches!(self, Self::TokenLoss) {
            self.scalar(&score)?;
            let targets = score["targets"]
                .as_u64()
                .ok_or("missing supervised tokens")?;
            let hits = score["correct"].as_u64().ok_or("missing correct tokens")?;
            if targets
                != rows
                    .iter()
                    .map(|r| r.targets.iter().filter(|&&y| y >= 0).count() as u64)
                    .sum::<u64>()
                || targets == 0
                || hits > targets
            {
                return Err("inconsistent dense loss denominator".into());
            }
            counts["loss"] = score["loss"].clone();
            counts["targets"] = json!(targets);
            counts["correct_tokens"] = json!(hits);
        }
        if rows.first().is_some_and(|r| r.decoder.is_some()) {
            counts["native_correct"] = json!(
                score["native_correct"]
                    .as_u64()
                    .ok_or("missing native token count")?
            );
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_window_uses_the_inclusive_ce_threshold_even_with_zero_exact_rows() -> Result<()> {
        let case =
            json!({"kind":"text","metric":"negative_loss","target":-1.63,"after_target":500});
        let scorer = SplitScorer::new(&case, None);
        let rule = SourceTarget::from_case(&case)?;
        for (loss, passed) in [(1.629, true), (1.63, true), (1.631, false)] {
            let scalar = scorer.scalar(&json!({"correct":0,"loss":loss}))?;
            assert_eq!(rule.met(0, 1024, scalar)?, passed);
        }
        for invalid in [
            json!({}),
            json!({"mse":1.62}),
            json!({"loss":-0.1}),
            json!({"loss":1.6,"mse":0.1}),
        ] {
            assert!(scorer.scalar(&invalid).is_err());
        }
        assert!(SplitScorer::Exact.scalar(&json!({"loss":1.63})).is_err());
        assert_eq!(rule.stop_after_crossing(2500, 250, 10000), 3000);
        Ok(())
    }
}

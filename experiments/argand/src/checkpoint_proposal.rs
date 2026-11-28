//! Test the ordinary checkpoint first. A failed Gaussian proposal restores the
//! exact weight image before the next gradient. Neither path changes moments.
use crate::{Result, engine::Engine, random::Random};
use serde_json::{Value, json};

pub(crate) fn add_gaussian(weights: &mut [f32], rng: &mut Random, stddev: f64) {
    for w in weights {
        *w += rng.gauss(0.0, stddev) as f32;
    }
}

fn perfect(score: &Value) -> Result<bool> {
    score["all_strict"]
        .as_bool()
        .ok_or_else(|| "missing strict checkpoint score".into())
}

pub(crate) fn evaluate(
    model: &mut Engine,
    rng: &mut Random,
    stddev: f64,
    mut score: impl FnMut(&mut Engine) -> Result<Value>,
) -> Result<Value> {
    let ordinary = score(model)?;
    if perfect(&ordinary)? {
        return Ok(
            json!({"accepted":true,"selected":"ordinary","ordinary":ordinary,"proposal":null}),
        );
    }
    let original = model.weights()?;
    let mut copy = original.clone();
    add_gaussian(&mut copy, rng, stddev);
    let attempted = model.restore(&copy).and_then(|()| {
        let proposal = score(model)?;
        Ok((perfect(&proposal)?, proposal))
    });
    match attempted {
        Ok((true, proposal)) => Ok(
            json!({"accepted":true,"selected":"proposal","ordinary":ordinary,"proposal":proposal}),
        ),
        result => {
            model.restore(&original)?;
            let (_, proposal) = result?;
            Ok(
                json!({"accepted":false,"selected":"ordinary","ordinary":ordinary,"proposal":proposal}),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::suite::Shape;

    #[test]
    #[ignore = "requires CUDA"]
    fn ordinary_success_skips_noise_and_only_a_certified_proposal_changes_weights() -> Result<()> {
        let shape = Shape {
            width: 16,
            heads: 2,
            layers: 1,
            vocab: 4,
            max_batch: 2,
            max_length: 3,
        };
        let mut model = Engine::new(shape, 1, 5)?;
        let original = model.weights()?;
        let mut rng = Random::new(371);
        let mut calls = 0;
        let ordinary = evaluate(&mut model, &mut rng, 1e-6, |_| {
            calls += 1;
            Ok(json!({"all_strict":true}))
        })?;
        assert_eq!(calls, 1);
        assert!(ordinary["proposal"].is_null());
        assert_eq!(original, model.weights()?);
        let rejected = evaluate(&mut model, &mut rng, 1e-6, |_| {
            Ok(json!({"all_strict":false}))
        })?;
        assert_eq!(rejected["accepted"], false);
        assert_eq!(original, model.weights()?);
        let failed = evaluate(&mut model, &mut rng, 1e-6, |m| {
            if m.weights()? == original {
                Ok(json!({"all_strict":false}))
            } else {
                Err("scorer error".into())
            }
        });
        assert!(failed.is_err());
        assert_eq!(original, model.weights()?);
        let accepted = evaluate(&mut model, &mut rng, 1e-6, |m| {
            Ok(json!({"all_strict":m.weights()? != original}))
        })?;
        assert_eq!(accepted["selected"], "proposal");
        assert_ne!(original, model.weights()?);
        Ok(())
    }
}

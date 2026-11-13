//! Only observed scores enter selection; the selector cannot inspect metrics.
use crate::*;
#[derive(Clone, Debug, PartialEq, Serialize)]
struct Score<'a> {
    reference: &'a str,
    train_ce: f64,
}
fn choose<'a>(scores: &'a [Score<'a>]) -> Option<&'a str> {
    let mut best = None;
    for score in scores {
        if best.is_none_or(|b: &Score<'_>| score.train_ce < b.train_ce - 1e-12) {
            best = Some(score);
        }
    }
    best.map(|b| b.reference)
}
pub(crate) fn validate_prior(suite: &Value, prior: &[Offer]) -> Result<()> {
    check(
        rows(suite, "tasks")?.len() == 35 && prior.len() == 105,
        "complete35-by3 comparison required",
    )?;
    let mut keys = BTreeSet::new();
    for offer in prior {
        task(suite, &offer.task)?;
        check(
            MODES.contains(&offer.mode.as_str()) && keys.insert((&offer.task, &offer.mode)),
            "missing or duplicate original task/mode",
        )?;
        check(
            offer.policy.is_none() && offer.transport.is_none(),
            "prior choice is not an initializer arm",
        )?;
        check(
            offer.train_ce.is_none_or(|v| v.is_finite() && v >= 0.),
            "invalid original CE",
        )?;
    }
    Ok(())
}
pub(crate) fn select(suite: &Value, prior: &[Offer], retained: &[Offer]) -> Result<Value> {
    validate_prior(suite, prior)?;
    check(retained.len() == 48, "every retained endpoint is required")?;
    let p = protocol()?;
    let eligible = p["retained_tasks"]
        .as_object()
        .ok_or("missing depth tasks")?;
    let mut unique = BTreeSet::new();
    for offer in retained {
        check(
            eligible.contains_key(&offer.task)
                && MODES.contains(&offer.mode.as_str())
                && offer
                    .policy
                    .as_deref()
                    .is_some_and(|v| POLICIES.contains(&v))
                && offer
                    .transport
                    .as_deref()
                    .is_some_and(|v| TRANSPORTS.contains(&v))
                && offer.train_ce.is_some_and(|v| v.is_finite() && v >= 0.)
                && unique.insert((&offer.task, &offer.mode, &offer.policy, &offer.transport)),
            "invalid or duplicate retained endpoint",
        )?;
    }
    let mut arms = vec![];
    for policy in POLICIES {
        let mut choices = vec![];
        for t in rows(suite, "tasks")? {
            let name = text(t, "name")?;
            for mode in MODES {
                let old = prior
                    .iter()
                    .find(|r| r.task == name && r.mode == mode)
                    .ok_or("missing prior mode")?;
                let mut scores = vec![];
                if let Some(ce) = old.train_ce {
                    scores.push(Score {
                        reference: &old.reference,
                        train_ce: ce,
                    });
                }
                if eligible.contains_key(name) {
                    for transport in TRANSPORTS {
                        let r = retained
                            .iter()
                            .find(|r| {
                                r.task == name
                                    && r.mode == mode
                                    && r.policy.as_deref() == Some(policy)
                                    && r.transport.as_deref() == Some(transport)
                            })
                            .ok_or("missing declared retained endpoint")?;
                        scores.push(Score {
                            reference: &r.reference,
                            train_ce: r.train_ce.ok_or("retained head missing CE")?,
                        });
                    }
                }
                choices.push(
                    json!({"task":name,"mode":mode,"prior_reference":old.reference,
                    "candidates":scores,"selected":choose(&scores)}),
                );
            }
        }
        arms.push(json!({"initializer":policy,"choices":choices}));
    }
    Ok(
        json!({"selection":"training_ce_then_earlier_order","tolerance":1e-12,
        "select_between_initializers":false,"heldout_inputs":false,"arms":arms}),
    )
}

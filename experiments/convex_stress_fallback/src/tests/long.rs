use super::*;

fn presence_request(context: usize, query: usize) -> causal::Request {
    let mut x = request();
    x.input.context = context;
    x.input.observations = vec![
        causal::Observation {
            tokens: vec![1, 1, query],
            label: 5,
        },
        causal::Observation {
            tokens: vec![0, 1, query],
            label: 4,
        },
    ];
    x
}

#[test]
fn long_supported_and_unsupported_prefixes_keep_original_causal_domain() -> Result<()> {
    let x = presence_request(5, 6);
    let head = candidate(&x)?;
    // The parent has a different query and a shorter training feature library.
    // Its original prediction function still accepts long causal prefixes.
    let prior = candidate(&presence_request(3, 5))?;
    let parent = Parent::Causal(prior.clone());
    assert_eq!(parent.shape(), (7, 3));
    assert_eq!(parent.prediction_limit(), 4096);
    let model = Model::new(head.clone(), parent, &x.input)?;
    assert_eq!(model.prediction_limit(), 4096);
    for length in [6, 65, 4096] {
        let mut prefix = vec![1; length];
        prefix[length / 2] = 0;
        prefix[length - 1] = 6;
        assert!(model.known(&prefix)?);
        assert_eq!(model.logits(&prefix)?, head.logits(&prefix));
        assert_eq!(winner(&model.logits(&prefix)?), 4);
        prefix[length - 1] = 5;
        assert!(!model.known(&prefix)?);
        assert_eq!(model.logits(&prefix)?, prior.logits(&prefix));
        assert_eq!(winner(&model.logits(&prefix)?), 4);
    }
    assert!(model.logits(&vec![1; 4097]).is_err());
    assert!(model.parent().logits(&vec![1; 4097]).is_err());
    let saved = serde_json::to_value(&model)?;
    let loaded = Model::load(saved, &x.input, model.binding())?;
    assert_eq!(loaded.prediction_limit(), 4096);
    assert_eq!(loaded.logits(&[1; 65])?, model.logits(&[1; 65])?);
    Ok(())
}

#[test]
fn longer_count_parent_domain_is_retained_without_expanding_its_value_alphabet() -> Result<()> {
    let x = presence_request(5, 6);
    let head = candidate(&x)?;
    let raw = count_value(7, 65);
    let old: convex_stress_counts::AdaptiveModel = serde_json::from_value(raw.clone())?;
    let previous = Parent::from_saved("counts", raw)?;
    let model = Model::new(head.clone(), previous, &x.input)?;
    assert_eq!(model.prediction_limit(), 65);
    let mut prefix = vec![1; 65];
    prefix[53] = 0;
    prefix[64] = 6;
    assert!(model.known(&prefix)?);
    assert_eq!(model.logits(&prefix)?, head.logits(&prefix));
    prefix[64] = 1;
    assert!(!model.known(&prefix)?);
    assert_eq!(model.logits(&prefix)?, old.logits(&prefix));
    assert!(model.logits(&vec![1; 66]).is_err());
    assert!(model.parent().logits(&vec![1; 66]).is_err());
    assert_eq!(model.parent().shape(), (7, 65));
    Ok(())
}

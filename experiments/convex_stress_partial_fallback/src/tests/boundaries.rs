use super::*;
fn presence(context: usize, query: usize) -> causal::Request {
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
fn partial_count_parent_has_three_actual_routes_without_domain_expansion() -> Result<()> {
    let x = presence(5, 6);
    let head = candidate(&x)?;
    let prior = parent(7, 5)?;
    let old = original::Model::new(head.clone(), prior.clone(), &x.input)?;
    let model = Model::new(head.clone(), prior.clone(), &x.input)?;
    assert_eq!(model.prediction_limit(), 4096);
    assert_eq!(model.parent().prediction_limit(), 5);
    assert_eq!(model.bounds(), old.bounds());
    assert_eq!(model.binding(), old.binding());
    assert_eq!(
        serde_json::to_value(&model)?["original"],
        serde_json::to_value(&old)?
    );
    let supported = [0, 1, 6];
    assert_eq!(
        model.route(&supported)?,
        Route {
            candidate_supported: true,
            parent_available: true,
            source: Source::CandidateSupported
        }
    );
    assert_eq!(model.logits(&supported)?, head.logits(&supported));
    let unsupported = [0, 1, 2];
    assert_eq!(
        model.route(&unsupported)?,
        Route {
            candidate_supported: false,
            parent_available: true,
            source: Source::Parent
        }
    );
    assert_eq!(model.logits(&unsupported)?, prior.logits(&unsupported)?);
    assert_ne!(model.logits(&unsupported)?, head.logits(&unsupported));
    let bytes = serde_json::to_vec(model.parent())?;
    for length in [6, 65, 4096] {
        let mut prefix = vec![1; length];
        prefix[length / 2] = 0;
        prefix[length - 1] = 6;
        assert_eq!(
            model.route(&prefix)?,
            Route {
                candidate_supported: true,
                parent_available: false,
                source: Source::CandidateSupported
            }
        );
        assert_eq!(model.logits(&prefix)?, head.logits(&prefix));
        assert!(old.logits(&prefix).is_err());
        prefix[length - 1] = 2;
        assert_eq!(
            model.route(&prefix)?,
            Route {
                candidate_supported: false,
                parent_available: false,
                source: Source::CandidateParentUnavailable
            }
        );
        assert_eq!(model.logits(&prefix)?, head.logits(&prefix));
        assert!(!model.known(&prefix)?);
        assert!(model.parent().logits(&prefix).is_err());
    }
    assert_eq!(serde_json::to_vec(model.parent())?, bytes);
    let extended_parent = Model::new(head.clone(), parent(7, 65)?, &x.input)?;
    let at_limit = vec![2; 65];
    assert_eq!(extended_parent.route(&at_limit)?.source, Source::Parent);
    assert_eq!(
        extended_parent.logits(&at_limit)?,
        extended_parent.parent().logits(&at_limit)?
    );
    let beyond = vec![2; 66];
    assert_eq!(
        extended_parent.route(&beyond)?.source,
        Source::CandidateParentUnavailable
    );
    assert_eq!(extended_parent.logits(&beyond)?, head.logits(&beyond));
    Ok(())
}

#[test]
fn causal_parent_keeps_its_actual_long_prefix_route_and_saved_state() -> Result<()> {
    let x = presence(5, 6);
    let head = candidate(&x)?;
    let prior = candidate(&presence(3, 5))?;
    let model = Model::new(head.clone(), Parent::Causal(prior.clone()), &x.input)?;
    assert_eq!(model.parent().shape(), (7, 3));
    assert_eq!(model.parent().prediction_limit(), 4096);
    for length in [6, 65, 4096] {
        let mut prefix = vec![1; length];
        prefix[length / 2] = 0;
        prefix[length - 1] = 6;
        assert_eq!(model.route(&prefix)?.source, Source::CandidateSupported);
        assert_eq!(model.logits(&prefix)?, head.logits(&prefix));
        prefix[length - 1] = 5;
        assert_eq!(model.route(&prefix)?.source, Source::Parent);
        assert_eq!(model.logits(&prefix)?, prior.logits(&prefix));
        assert_ne!(model.logits(&prefix)?, head.logits(&prefix));
    }
    let restored = Model::load(serde_json::to_value(&model)?, &x.input, model.binding())?;
    assert_eq!(restored.route(&[2; 65])?, model.route(&[2; 65])?);
    assert_eq!(restored.logits(&[2; 65])?, model.logits(&[2; 65])?);
    Ok(())
}

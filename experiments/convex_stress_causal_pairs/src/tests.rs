use crate::*;
pub(crate) fn pair_request() -> Request {
    Request {
        input: Input {
            vocabulary: 7,
            context: 5,
            observations: vec![
                Observation {
                    tokens: vec![0, 0, 6],
                    label: 2,
                },
                Observation {
                    tokens: vec![0, 1, 6],
                    label: 3,
                },
                Observation {
                    tokens: vec![1, 0, 6],
                    label: 4,
                },
                Observation {
                    tokens: vec![1, 1, 6],
                    label: 5,
                },
            ],
        },
        policy: Policy::default(),
    }
}
#[test]
fn cached_features_match_independent_prefix_semantics_exhaustively() {
    let library = features::library(3, 5);
    for length in 1..=5 {
        for code in 0..3usize.pow(length as u32) {
            let mut state = code;
            let tokens: Vec<_> = (0..length)
                .map(|_| {
                    let t = state % 3;
                    state /= 3;
                    t
                })
                .collect();
            let expected: Vec<_> = library
                .iter()
                .map(|&f| features::value(f, 3, &tokens))
                .collect();
            assert_eq!(features::encode(3, 5, &tokens), expected);
        }
    }
    let a = features::value(
        features::Feature::Successor { token: 1, hop: 1 },
        3,
        &[1, 2, 1],
    );
    assert_eq!(a, 2); // The latest incomplete occurrence must not erase its predecessor.
}
#[test]
fn lazy_catalogue_contains_every_pair_once_and_prioritizes_presence() {
    for (v, n) in [(2, 1), (3, 5), (7, 8)] {
        let keys = features::library(v, n).len();
        let all: Vec<_> = features::catalogue(v, n).collect();
        assert_eq!(all.len(), keys + keys * (keys - 1) / 2);
        for (i, c) in all.iter().take(keys).enumerate() {
            assert_eq!(*c, features::Candidate::Single(i));
        }
        let mut pairs = std::collections::BTreeSet::new();
        for c in all.iter().skip(keys) {
            if let features::Candidate::Pair(a, b) = *c {
                assert!(a < b);
                assert!(pairs.insert((a, b)));
            } else {
                panic!("single after pairs");
            }
        }
        assert_eq!(features::library(v, n)[0], features::Feature::Presence(0));
    }
}
#[test]
fn raw_four_class_pair_defeats_every_single_feature_and_replays() -> Result<()> {
    let request = pair_request();
    let report = discovery::run(request.clone())?;
    discovery::replay(&request, &report)?;
    assert!(report.reason.is_none());
    let group = &report.groups[0];
    let lag_start = 2 * request.input.vocabulary + 1;
    assert_eq!(
        group.decoder.as_ref().unwrap().candidate,
        features::Candidate::Pair(lag_start + 1, lag_start + 2)
    );
    assert!(
        group
            .attempts
            .iter()
            .take(features::library(7, 5).len())
            .all(|a| matches!(a.decision, discovery::Decision::Conflict { .. }))
    );
    assert_eq!(group.decoder.as_ref().unwrap().state([2, 0]), None);
    assert!(!report.unseen_identification);
    let saved = serde_json::to_vec(&report)?;
    assert_eq!(serde_json::from_slice::<discovery::Report>(&saved)?, report);
    Ok(())
}
#[test]
fn invariant_prior_uses_presence_and_generalizes_the_fixture_shift() -> Result<()> {
    let mut request = pair_request();
    request.input.observations = vec![
        Observation {
            tokens: vec![1, 1, 6],
            label: 5,
        },
        Observation {
            tokens: vec![0, 1, 6],
            label: 4,
        },
    ];
    let report = discovery::run(request.clone())?;
    discovery::replay(&request, &report)?;
    let decoder = report.groups[0].decoder.as_ref().unwrap();
    assert_eq!(decoder.candidate, features::Candidate::Single(0));
    let library = features::library(7, 5);
    let value = decoder.candidate.original(&library, 7, &[1, 0, 6]);
    assert_eq!(decoder.labels[decoder.state(value).unwrap()], 4);
    // The same two observed labels also fit first-position routing, but that
    // candidate predicts a different answer after the shift. The prior matters.
    assert_ne!(request.input.observations[0].label, 4);
    Ok(())
}
#[test]
fn bounded_failures_retain_traces_and_altered_evidence_is_rejected() -> Result<()> {
    let original = pair_request();
    let report = discovery::run(original.clone())?;
    for i in 0..5 {
        let mut bad = report.clone();
        match i {
            0 => bad.groups[0].decoder.as_mut().unwrap().labels[0] = 0,
            1 => {
                bad.groups[0].attempts[0].decision = discovery::Decision::Conflict {
                    first: 0,
                    second: 0,
                }
            }
            2 => bad.work.visits += 1,
            3 => bad.unseen_identification = true,
            _ => bad.groups[0].attempts.swap(0, 1),
        }
        assert!(discovery::replay(&original, &bad).is_err());
    }
    for i in 0..4 {
        let mut capped = original.clone();
        match i {
            0 => capped.policy.visits = 0,
            1 => capped.policy.attempts = 0,
            2 => capped.policy.feature_cells = 0,
            _ => capped.policy.states_per_query = 1,
        }
        let out = discovery::run(capped.clone())?;
        assert!(out.reason.is_some());
        discovery::replay(&capped, &out)?;
    }
    let mut conflict = original;
    conflict.input.observations.push(Observation {
        tokens: vec![0, 0, 6],
        label: 0,
    });
    let out = discovery::run(conflict.clone())?;
    assert!(out.reason.is_some());
    discovery::replay(&conflict, &out)?;
    Ok(())
}

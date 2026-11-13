use super::*;

pub(crate) fn fixture() -> (Input, Vec<[usize; 2]>) {
    let (n, q) = (17, 7);
    let mut targets = vec![];
    let mut observations = vec![];
    for j in 0..2 * q {
        for i in 0..2 * q {
            let row = 1 + (5 * i) % (2 * q);
            let column = 1 + (3 * j) % (2 * q);
            let phase = (i + 3 * j + 1) % q;
            let alias = usize::from((131 * i + 17 * j + 7 * i * j) % 5 < 2);
            let label = 5 + (5 * (2 * phase + alias) + 3) % (2 * q);
            let pair = [row + n * column, label];
            targets.push(pair);
            if (i + 3 * j) % 5 != 0 {
                observations.push(pair);
            }
        }
    }
    (
        Input {
            n,
            vocabulary: 29,
            observations,
            policy: Policy::default(),
        },
        targets,
    )
}
#[test]
fn fixed_catalogue_discovers_colliding_codes_and_predicts_unseen_codes() -> Result<()> {
    let (input, targets) = fixture();
    let mut admitted = vec![];
    for prime in catalogue().into_iter().filter(|&p| p <= 11) {
        let request = Request {
            input: input.clone(),
            prime,
        };
        let out = run(request.clone())?;
        check(&request, &out)?;
        replay(&request, &out)?;
        if let Some(code) = &out.code {
            admitted.push(prime);
            assert_eq!(code.names.classes.len(), 14);
            assert_eq!(out.code_fibers.iter().filter(|f| f.len() == 2).count(), 7);
            assert!(!out.raw_label_identification);
            assert!(targets.len() > input.observations.len());
            for &[cell, label] in &targets {
                assert_eq!(
                    code.query(cell % input.n, cell / input.n),
                    code.label(label)
                );
            }
            assert_eq!(code.query(0, 1), None);
            assert_eq!(code.query(1, 16), None);
            assert_eq!(code.label(0), None);
        }
    }
    assert_eq!(admitted, vec![7]);
    assert_eq!(catalogue().last(), Some(&257));
    Ok(())
}
#[test]
fn class_axis_names_and_observation_order_are_only_categorical() -> Result<()> {
    let (input, targets) = fixture();
    let row = |i| (3 * i + 2) % input.n;
    let column = |j| (5 * j + 4) % input.n;
    let label = |c| (2 * c + 7) % input.vocabulary;
    let map = |pair: &[usize; 2]| {
        [
            row(pair[0] % input.n) + input.n * column(pair[0] / input.n),
            label(pair[1]),
        ]
    };
    let mut renamed = input.clone();
    renamed.observations = input.observations.iter().rev().map(map).collect();
    let request = Request {
        input: renamed,
        prime: 7,
    };
    let out = run(request.clone())?;
    check(&request, &out)?;
    let code = out.code.as_ref().ok_or("renamed fixture refused")?;
    for pair in &targets {
        let [cell, label] = map(pair);
        assert_eq!(
            code.query(cell % input.n, cell / input.n),
            code.label(label)
        );
    }
    Ok(())
}
#[test]
fn capped_ambiguous_and_bad_inputs_preserve_explicit_refusals() -> Result<()> {
    let (input, _) = fixture();
    let mut request = Request { input, prime: 7 };
    request.input.policy.elimination.scalar_updates = 0;
    let capped = run(request.clone())?;
    check(&request, &capped)?;
    replay(&request, &capped)?;
    assert!(capped.code.is_none());
    assert!(capped.field.space.is_none());
    request.input.policy = Policy::default();
    request.input.policy.equations = 1;
    let ambiguous = run(request.clone())?;
    check(&request, &ambiguous)?;
    assert!(ambiguous.code.is_none());
    assert!(ambiguous.field.free.len() > 1);
    request
        .input
        .observations
        .push(request.input.observations[0]);
    assert!(run(request.clone()).is_err());
    request.input.observations.pop();
    request.prime = 6;
    assert!(run(request).is_err());
    Ok(())
}
#[test]
fn every_original_equation_is_checked_beyond_the_discovery_prefix() -> Result<()> {
    let (mut input, targets) = fixture();
    // Full fixture observations precede a previously unseen cell with a
    // deliberately incompatible raw class. It is excluded from elimination.
    input.policy.equations = input.observations.len();
    let missing = targets
        .iter()
        .find(|pair| !input.observations.iter().any(|s| s[0] == pair[0]))
        .unwrap();
    let clean = Request {
        input: input.clone(),
        prime: 7,
    };
    let code = run(clean)?.code.ok_or("base fixture refused")?;
    let wrong = *code
        .names
        .classes
        .iter()
        .find(|&&c| code.label(c) != code.label(missing[1]))
        .unwrap();
    input.observations.push([missing[0], wrong]);
    let request = Request { input, prime: 7 };
    let out = run(request.clone())?;
    check(&request, &out)?;
    replay(&request, &out)?;
    assert_eq!(out.reason, "original_observation_conflict");
    assert_eq!(out.conflict, Some(request.input.observations.len() - 1));
    assert!(out.code.is_none());
    Ok(())
}
#[test]
fn mutated_codes_fields_work_names_and_claims_fail_independent_checks() -> Result<()> {
    let (input, _) = fixture();
    let request = Request { input, prime: 7 };
    let out = run(request.clone())?;
    let json = serde_json::to_vec(&out)?;
    assert_eq!(serde_json::from_slice::<Report>(&json)?, out);
    let mut mutations = vec![];
    let mut bad = out.clone();
    bad.code.as_mut().unwrap().classes[0] = 6;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.names.rows.swap(0, 1);
    mutations.push(bad);
    let mut bad = out.clone();
    bad.field.augmented[0][0] = (bad.field.augmented[0][0] + 1) % 7;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.work.observation_checks -= 1;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.raw_label_identification = true;
    mutations.push(bad);
    let mut bad = out.clone();
    bad.code_fibers[0].clear();
    mutations.push(bad);
    let mut bad = out.clone();
    bad.inverse = Some(0);
    mutations.push(bad);
    for bad in mutations {
        assert!(check(&request, &bad).is_err());
    }
    Ok(())
}

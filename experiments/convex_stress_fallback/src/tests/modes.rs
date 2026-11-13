use super::*;

#[test]
fn all_saved_modes_retain_their_candidate_trace_and_frozen_parent() -> Result<()> {
    use causal::training::{Mode, Policy};
    let mut x = request();
    x.input.observations.push(x.input.observations[0].clone());
    let mut initial = candidate(&x)?;
    for w in &mut initial.coefficients {
        *w *= 0.02;
    }
    let previous = parent(7, 5)?;
    let old_bytes = serde_json::to_vec(&previous)?;
    let unknown = [0, 2, 6];
    let expected = previous.logits(&unknown)?;
    for policy in [
        Policy {
            mode: Mode::Full,
            updates: 1,
            batch: 2,
            learning_rate: 0.1,
        },
        Policy {
            mode: Mode::Mini,
            updates: 3,
            batch: 2,
            learning_rate: 0.1,
        },
        Policy {
            mode: Mode::Forward,
            updates: 0,
            batch: 2,
            learning_rate: 0.1,
        },
    ] {
        let out = causal::training::run(&initial, &x.input, policy)?;
        causal::training::replay(&initial, &x.input, &out)?;
        let wrapped = Model::new(out.model.clone(), previous.clone(), &x.input)?;
        assert_eq!(wrapped.logits(&unknown)?, expected);
        assert_eq!(serde_json::to_vec(wrapped.parent())?, old_bytes);
        assert_eq!(
            batch_loss(&wrapped, &x.input, &[0, 1, 2, 3, 4])?,
            out.final_ce
        );
        if out.policy.mode == Mode::Forward {
            assert_eq!(out.work.gradient_batches, 0);
            assert_eq!(out.work.scalar_derivative_terms, 0);
            assert_eq!(out.model, initial);
        } else {
            assert_eq!(out.work.gradient_batches, out.policy.updates);
            assert_ne!(out.model, initial);
        }
    }
    // An epoch with a shorter tail uses the actual original batch lengths.
    // Calling the old full-batch step on each supplied batch retains its
    // normalization; it does not claim the old cyclic Mini runner has a tail.
    let mut current = initial;
    let mut examples = 0;
    for ids in [vec![0, 1], vec![2, 3], vec![4]] {
        let batch = Input {
            vocabulary: 7,
            context: 5,
            observations: ids
                .iter()
                .map(|&i| x.input.observations[i].clone())
                .collect(),
        };
        let out = causal::training::run(
            &current,
            &batch,
            Policy {
                mode: Mode::Full,
                updates: 1,
                batch: 2,
                learning_rate: 0.1,
            },
        )?;
        assert_eq!(out.steps[0].count, ids.len());
        examples += out.work.gradient_examples;
        current = out.model;
        let wrapped = Model::new(current.clone(), previous.clone(), &x.input)?;
        assert_eq!(wrapped.logits(&unknown)?, expected);
    }
    assert_eq!(examples, 5);
    Ok(())
}

#[test]
fn typed_saved_parent_dispatch_and_exact_budget_are_checked() -> Result<()> {
    let raw = count_value(7, 5);
    let checkpoint = json!({"kind":"Counts","model":raw});
    let loaded = Parent::from_checkpoint(checkpoint)?;
    assert!(Parent::from_checkpoint(json!({"kind":"Moments","model":{}})).is_err());
    assert!(
        Parent::from_checkpoint(json!({"kind":"Counts","model":count_value(7,5),"predictions":[]}))
            .is_err()
    );
    let Parent::Counts(counts) = loaded else {
        panic!("wrong saved kind")
    };
    let norms = counts.column_l1()?;
    let bounds = counts.validate()?;
    assert!(norms.iter().any(|&n| n > 0.));
    assert!(norms.iter().all(|&n| n <= bounds.radius));
    // Changing imaginary candidate coefficients changes the actual prediction.
    let x = request();
    let mut head = candidate(&x)?;
    let previous = parent(7, 5)?;
    let before = Model::new(head.clone(), previous.clone(), &x.input)?;
    head.coefficients[3 * 7 + 3] += 0.5;
    let after = Model::new(head, previous, &x.input)?;
    assert_ne!(before.logits(&[0, 1, 6])?, after.logits(&[0, 1, 6])?);
    Ok(())
}

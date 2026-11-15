//! Original nonlinear fixture from signed_compare at9c7370a5, extended by the
//! actual signed component. Only synthetic observed pairs enter construction.
use super::*;
use crate::old as head;
use crate::old::records;
use convex_stress_shared_roles::training;
use head::kernel;

pub(crate) fn fixture(mode: head::Mode) -> Result<(Value, data::Task, Value)> {
    with_prime(mode, 3)
}
pub(crate) fn with_prime(mode: head::Mode, prime: u16) -> Result<(Value, data::Task, Value)> {
    // Use the diagonal and upper band of the reviewed316 fixture. Removing
    // the symmetric lower-band constraints leaves unmatched latent blocks,
    // so the fixed coverage fallback is actually exercised.
    let observations: Vec<_> = (0usize..49)
        .filter(|cell| cell % 7 == cell / 7 || cell % 7 == cell / 7 + 1)
        .map(|cell| [cell, if cell % 7 == cell / 7 { 2 } else { 0 }])
        .collect();
    let report = kernel::discovery::run(kernel::discovery::Request {
        prime,
        input: kernel::discovery::Input {
            n: 7,
            vocabulary: 7,
            observations: observations.clone(),
            policy: kernel::discovery::Policy::default(),
        },
    })?;
    let initial = kernel::initialize(&report)?.model;
    let records: Vec<_> = (0..45)
        .map(|id| {
            let [cell, label] = observations[id % observations.len()];
            records::Record { id, cell, label }
        })
        .collect();
    let policy = training::policies(records.len(), 8)?
        .into_iter()
        .find(|p| {
            matches!(
                (mode, p.mode),
                (head::Mode::Full, training::Mode::Full)
                    | (head::Mode::Mini, training::Mode::Mini)
                    | (head::Mode::Forward, training::Mode::Forward)
            )
        })
        .unwrap();
    let outcome = training::run(&initial, &records, policy)?;
    training::replay(&initial, &records, &outcome)?;
    let updates = outcome.steps.len();
    assert_eq!(
        updates,
        match mode {
            head::Mode::Full => 1,
            head::Mode::Mini => 6,
            head::Mode::Forward => 0,
        }
    );
    if mode == head::Mode::Mini {
        assert_eq!(outcome.steps.last().unwrap().count, 5);
    }
    let source = outcome.model;
    let pin = head::digest(&source)?;
    let context = head::regions::parent::scale::context::Model::new(source, records.clone(), &pin)?;
    let pin = head::digest(&context)?;
    let scaled = head::regions::parent::scale::Model::new(context, &pin)?;
    let pin = head::digest(&scaled)?;
    let endpoint = head::Endpoint {
        mode,
        source_reference: "context-scale:context:fixture".into(),
        source_gradient_training_run: mode != head::Mode::Forward,
        source_optimizer_updates: Some(updates),
        mixture_batch: 8,
    };
    let mixture = head::regions::parent::parent::Model::new(scaled, endpoint, &pin)?;
    let pin = head::digest(&mixture)?;
    let parent = head::regions::parent::Model::new(mixture, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::regions::Model::new(parent, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::parent::parent::Model::new(parent, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::parent::Model::new(parent, &pin)?;
    let pin = head::digest(&parent)?;
    let parent = head::Model::new(parent, &pin)?;
    let mode = match mode {
        head::Mode::Full => "full_batch",
        head::Mode::Mini => "minibatch",
        head::Mode::Forward => "forward_only",
    };
    let example = |cell: usize, label: usize| data::Example {
        task: "fixture".into(),
        tokens: vec![(cell / 7) as i64, (cell % 7) as i64],
        targets: vec![-100, label as i64],
        changes: vec![false, false],
        prompt: vec![],
        answer: vec![],
        limit: 0,
        field: None,
        row_index: None,
    };
    let task = data::Task {
        vocab: 7,
        selection: "validation".into(),
        splits: BTreeMap::from([
            (
                "train".into(),
                records.iter().map(|r| example(r.cell, r.label)).collect(),
            ),
            (
                "validation".into(),
                (0..49)
                    .map(|cell| example(cell, if cell % 7 == cell / 7 { 2 } else { 0 }))
                    .collect(),
            ),
        ]),
    };
    Ok((
        json!({"prior":parent,"supplement":null}),
        task,
        json!({"reference":"crossing-exclusion:raw-exclusion:raw-regions:gated-context:context-mixture:context-scale:context:fixture","training_mode":mode}),
    ))
}

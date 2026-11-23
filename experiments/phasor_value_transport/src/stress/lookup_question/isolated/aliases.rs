//! Attribution of the largest learned coefficient to the distinct role tuples
//! sharing its bucket. The frequency estimate sees only training prompts.
use super::*;
use std::collections::BTreeSet;

pub(in crate::stress) struct Audit {
    pub predictions: Vec<Vec<Option<i64>>>,
    pub details: Value,
}

pub(in crate::stress) fn run(
    cfg: &Config,
    weights: &[f32],
    training: &[Example],
    examples: &[Example],
    budgets: &[usize],
    refined: bool,
    walk: crate::typed_lookup::Walk,
) -> Result<Audit> {
    let geometry = Geometry::new(cfg, weights);
    let path = cfg
        .head
        .peer_read()
        .and_then(|r| r.path)
        .ok_or("no discrete path read")?;
    let raw = &weights[coefficients::table_range(cfg)?];
    let maximum = raw.iter().copied().map(f32::abs).fold(0., f32::max);
    if maximum == 0. {
        return Err("the trained path table is identically zero".into());
    }
    let bucket = raw
        .iter()
        .position(|w| w.abs() == maximum)
        .ok_or("empty coefficient table")?;
    let tied = raw.iter().filter(|w| w.abs() == maximum).count();
    // The common power gain preserves the leading coefficient, including its sign.
    let strength = raw[bucket];
    let keys = |e: &Example| {
        e.prompt
            .iter()
            .map(|&id| geometry.addresses[id as usize])
            .collect::<Vec<_>>()
    };
    let mut counts = BTreeMap::new();
    for example in training {
        let mut seen = BTreeSet::new();
        crate::typed_lookup::visit_final_bucket(
            &keys(example),
            &[path.passes],
            path.ordered,
            bucket,
            walk,
            |_, roles, _, _| {
                seen.insert(roles);
            },
        );
        for roles in seen {
            *counts.entry(roles).or_insert(0usize) += 1;
        }
    }
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_unstable_by_key(|(roles, n)| (std::cmp::Reverse(*n), *roles));
    let &(selected, frequency) = ranked
        .first()
        .ok_or("the winning bucket has no training witness")?;
    let mut identities = vec![BTreeSet::new(); budgets.len()];
    let mut different = vec![0usize; budgets.len()];
    let mut absent = vec![0usize; budgets.len()];
    let mut predictions = vec![Vec::with_capacity(examples.len()); budgets.len()];
    for example in examples {
        let mut seen = vec![BTreeSet::new(); budgets.len()];
        let mut votes = vec![vec![0.; example.prompt.len()]; budgets.len()];
        crate::typed_lookup::visit_final_bucket(
            &keys(example),
            budgets,
            path.ordered,
            bucket,
            walk,
            |i, roles, pos, sign| {
                seen[i].insert(roles);
                if !refined || roles == selected {
                    votes[i][pos] += sign * strength;
                }
            },
        );
        for (i, (roles, v)) in seen.into_iter().zip(votes).enumerate() {
            different[i] += usize::from(roles.iter().any(|&r| r != selected));
            absent[i] += usize::from(!roles.contains(&selected));
            identities[i].extend(roles);
            predictions[i].push(Some(geometry.project(&example.prompt, &v).0 as i64));
        }
    }
    let by_budget: BTreeMap<_, _> = budgets
        .iter()
        .enumerate()
        .map(|(i, &b)| {
            (
                b,
                json!({
                    "distinct_tuples":identities[i].len(),"prompts_with_other_tuples":different[i],
                    "prompts_without_selected":absent[i]
                }),
            )
        })
        .collect();
    let distinct = geometry
        .addresses
        .iter()
        .flatten()
        .collect::<BTreeSet<_>>()
        .len();
    Ok(Audit {
        predictions,
        details: json!({
            "selected_by":"training prompt presence, ties by lexicographic role tuple",
            "training_prompts":training.len(),"training_tuples":ranked.len(),"selected_training_presence":frequency,
            "largest_presence_counts":ranked.iter().take(5).map(|r|r.1).collect::<Vec<_>>(),
            "selected_bucket":bucket,"maximum_bucket_ties":tied,"selected_role_addresses":selected,
            "vocabulary":cfg.vocab,"distinct_vocabulary_addresses":distinct,
            "missing_vocabulary_addresses":geometry.addresses.iter().filter(|a|a.is_none()).count(),
            "evaluation_prompts":examples.len(),"by_budget":by_budget
        }),
    })
}

//! Audit runs after the constructor's label-free results have been hashed.
use crate::{
    data::Split,
    model::{Forward, P, W},
    transport::{self, Constructed, Grouping, METHODS, summary},
};
use serde_json::{Value, json};

fn winner(row: &[f64]) -> usize {
    let mut best = 0;
    for v in 1..row.len() {
        if row[v] > row[best] {
            best = v;
        }
    }
    best
}
fn target(q: usize) -> usize {
    (q / P + q % P).pow(3) % P
}
fn correct(out: &Forward, indices: &[usize]) -> usize {
    indices
        .iter()
        .filter(|&&q| winner(&out.logits[q * P..(q + 1) * P]) == target(q))
        .count()
}

fn audit(c: &Constructed, out: &Forward, split: &Split) -> Value {
    let part = |indices: &[usize]| {
        let mut methods = serde_json::Map::new();
        for (m, name) in METHODS.iter().enumerate() {
            let mut positive = 0;
            let mut wrong = 0;
            let mut minimums = Vec::new();
            let mut execution_budgets = Vec::new();
            let mut worst_violation = 0f64;
            for &q in indices {
                if let Some(label) = c.labels[q] {
                    assert_eq!(label, target(q), "route failed to transfer the true label");
                    let lower = c.lower[m][q].unwrap();
                    minimums.push(lower);
                    let actual = (0..P)
                        .filter(|&v| v != label)
                        .map(|v| out.raw[q * P + label] - out.raw[q * P + v])
                        .fold(f64::INFINITY, f64::min);
                    worst_violation = worst_violation.max(lower - actual);
                    if lower > 0. {
                        positive += 1;
                        wrong += usize::from(winner(&out.logits[q * P..(q + 1) * P]) != label);
                        execution_budgets.push(lower / (2. * out.denominators[q]));
                    }
                }
            }
            assert!(
                worst_violation < 1e-8,
                "transport margin lower bound violated"
            );
            assert_eq!(
                wrong, 0,
                "a positive condition disagrees with the source classifier"
            );
            methods.insert(
                (*name).to_owned(),
                json!({"positive":positive,"positive_but_incorrect":wrong,
                "lower_margin":summary(&minimums),"max_numerical_bound_violation":worst_violation,
                "remaining_per_logit_error_budget":summary(&execution_budgets)}),
            );
        }
        json!({"count":indices.len(),"source_correct":correct(out,indices),
        "residual_only_correct":indices.iter().filter(|&&q|winner(&out.residual[q*P..(q+1)*P])==target(q)).count(),
            "missing_anchor":indices.iter().filter(|&&q|c.labels[q].is_none()).count(),"methods":methods})
    };
    json!({"train":part(&split.train),"validation":part(&split.validation),"groups":c.groups})
}

pub fn analyze(out: &Forward, split: &Split, weights: &[f64]) -> Value {
    let inputs: Vec<_> = (0..P * P).map(|q| (q / P, q % P)).collect();
    // Only observed labels are made available to construct(). Query truth enters audit() later.
    let training: Vec<_> = split.train.iter().map(|&q| (q, target(q))).collect();
    let sum = transport::construct(
        &inputs,
        &out.states,
        &weights[..P * W],
        &training,
        W,
        P,
        Grouping::Sum,
    );
    let orbit = transport::construct(
        &inputs,
        &out.states,
        &weights[..P * W],
        &training,
        W,
        P,
        Grouping::Orbit,
    );
    let frozen = json!({"sum":sum.frozen(),"orbit":orbit.frozen()});
    let construction_sha256 =
        crate::hash(&serde_json::to_vec(&frozen).expect("serializable construction"));
    // Both constructions are complete and hashed before any query label/output is audited.
    let sum_report = audit(&sum, out, split);
    let orbit_report = audit(&orbit, out, split);
    let part = |indices: &[usize]| {
        json!({"count":indices.len(),"source_correct":correct(out,indices),
        "final_rms_denominator":summary(&indices.iter().map(|&q|out.denominators[q]).collect::<Vec<_>>())})
    };
    json!({"construction_sha256":construction_sha256,"sum":sum_report,"orbit":orbit_report,
        "train":part(&split.train),"validation":part(&split.validation),
        "status":"Training-label and query-state construction; query targets/logits only audited afterward; FP64, eta=0"})
}

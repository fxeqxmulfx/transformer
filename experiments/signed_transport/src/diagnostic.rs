//! Query targets and scores are audited after both geometric constructions are hashed.
use crate::{
    data::Split,
    model::{Forward, P, W},
    transport::{self, Constructed, Grouping, summary},
};
use serde_json::{Value, json};

fn target(q: usize) -> usize {
    (q / P + q % P).pow(3) % P
}
fn winner(row: &[f64]) -> usize {
    (1..row.len()).fold(0, |best, c| if row[c] > row[best] { c } else { best })
}
fn correct(out: &Forward, indices: &[usize]) -> usize {
    indices
        .iter()
        .filter(|&&q| winner(&out.logits[q * P..(q + 1) * P]) == target(q))
        .count()
}
fn actual_loss(out: &Forward, q: usize, label: usize) -> f64 {
    let wrong: Vec<_> = (0..P)
        .filter(|&c| c != label)
        .map(|c| out.logits[q * P + c] - out.logits[q * P + label])
        .collect();
    transport::softplus(transport::log_sum_exp(&wrong))
}
fn audit(c: &Constructed, out: &Forward, split: &Split) -> Value {
    let part = |indices: &[usize]| {
        let actual: Vec<_> = indices
            .iter()
            .filter_map(|&q| {
                c.labels[q].map(|label| {
                    assert_eq!(label, target(q), "training route supplied wrong task label");
                    let loss = actual_loss(out, q, label);
                    let gap = (0..P)
                        .filter(|&v| v != label)
                        .map(|v| out.logits[q * P + label] - out.logits[q * P + v])
                        .fold(f64::INFINITY, f64::min);
                    let incorrect = winner(&out.logits[q * P..(q + 1) * P]) != label;
                    (q, loss, (-loss).exp(), gap, incorrect)
                })
            })
            .collect();
        let mut methods = serde_json::Map::new();
        for (m, name) in c.methods.iter().enumerate() {
            let (mut certified, mut argmax, mut wrong, mut wrong_argmax) = (0, 0, 0, 0);
            let (mut probability_violation, mut loss_violation, mut gap_violation) =
                (0f64, 0f64, 0f64);
            let (mut floors, mut ceilings, mut gaps, mut log_odds, mut budgets) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for &(q, loss, probability, gap, incorrect) in &actual {
                let b = c.bounds[m][q].unwrap();
                probability_violation = probability_violation.max(b.probability - probability);
                loss_violation = loss_violation.max(loss - b.loss);
                gap_violation = gap_violation.max(b.gap - gap);
                floors.push(b.probability);
                ceilings.push(b.loss);
                gaps.push(b.gap);
                log_odds.push(b.log_odds);
                if b.log_odds < 0. {
                    certified += 1;
                    wrong += usize::from(incorrect);
                    budgets.push(-b.log_odds / 2.);
                }
                if b.gap > 0. {
                    argmax += 1;
                    wrong_argmax += usize::from(incorrect);
                }
            }
            assert!(
                probability_violation < 1e-8 && loss_violation < 1e-8 && gap_violation < 1e-8,
                "signed transport bound violated"
            );
            assert_eq!(
                wrong + wrong_argmax,
                0,
                "positive condition disagrees with source"
            );
            assert!(certified <= argmax);
            let full_mean = if ceilings.len() == indices.len() {
                Some(ceilings.iter().sum::<f64>() / indices.len() as f64)
            } else {
                None
            };
            methods.insert(name.clone(),json!({"probability_certified":certified,"argmax_certified":argmax,
                "probability_certified_but_incorrect":wrong,"argmax_certified_but_incorrect":wrong_argmax,
                "probability_lower":summary(&floors),"ce_upper":summary(&ceilings),"gap_lower":summary(&gaps),
                "log_odds_upper":summary(&log_odds),"remaining_per_logit_error_budget":summary(&budgets),
                "full_split_mean_ce_upper":full_mean,"max_probability_bound_violation":probability_violation,
                "max_ce_bound_violation":loss_violation,"max_gap_bound_violation":gap_violation}));
        }
        json!({"count":indices.len(),"source_correct":correct(out,indices),"missing_reference":indices.len()-actual.len(),"methods":methods})
    };
    json!({"train":part(&split.train),"validation":part(&split.validation),"groups":c.groups})
}

pub fn analyze(out: &Forward, split: &Split, weights: &[f64]) -> Value {
    let inputs: Vec<_> = (0..P * P).map(|q| (q / P, q % P)).collect();
    let states: Vec<_> = out
        .states
        .as_chunks::<W>()
        .0
        .iter()
        .enumerate()
        .flat_map(|(q, row)| row.iter().map(move |v| v / out.denominators[q]))
        .collect();
    let training: Vec<_> = split.train.iter().map(|&q| (q, target(q))).collect();
    let sum = transport::construct(
        &inputs,
        &states,
        &weights[..P * W],
        &training,
        W,
        P,
        Grouping::Sum,
    );
    let orbit = transport::construct(
        &inputs,
        &states,
        &weights[..P * W],
        &training,
        W,
        P,
        Grouping::Orbit,
    );
    let frozen = json!({"sum":sum.frozen(),"orbit":orbit.frozen()});
    let construction_sha256 =
        crate::hash(&serde_json::to_vec(&frozen).expect("finite serializable construction"));
    // No held-out target/score audit precedes the construction hash.
    let sum_report = audit(&sum, out, split);
    let orbit_report = audit(&orbit, out, split);
    let part = |indices: &[usize]| {
        json!({"count":indices.len(),"source_correct":correct(out,indices),
        "mean_cross_entropy":indices.iter().map(|&q|actual_loss(out,q,target(q))).sum::<f64>()/indices.len() as f64,
        "residual_only_correct":indices.iter().filter(|&&q|winner(&out.residual[q*P..(q+1)*P])==target(q)).count(),
        "raw_margin":summary(&indices.iter().map(|&q|(0..P).filter(|&v|v!=target(q)).map(|v|out.raw[q*P+target(q)]-out.raw[q*P+v]).fold(f64::INFINITY,f64::min)).collect::<Vec<_>>()),
        "final_rms_denominator":summary(&indices.iter().map(|&q|out.denominators[q]).collect::<Vec<_>>())})
    };
    json!({"construction_sha256":construction_sha256,"train":part(&split.train),"validation":part(&split.validation),
        "sum":sum_report,"orbit":orbit_report,"status":"Training-only subspaces and signed scores; unlabeled query geometry; FP64 eta=0; training queries use opposite folds"})
}

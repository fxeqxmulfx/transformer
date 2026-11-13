use super::*;

pub(super) fn request(
    parent: &search::Report,
    index: usize,
    policy: &Policy,
) -> Result<categorical::Request> {
    let c = parent
        .candidates
        .get(index)
        .ok_or("missing original parent candidate")?;
    let a = parent
        .attempts
        .get(c.view)
        .ok_or("missing original categorical view")?;
    if a.decision != (search::Decision::Accepted { candidate: index }) {
        return Err("candidate is not the admitted endpoint of its original view".into());
    }
    let code = a
        .coding
        .as_ref()
        .ok_or("accepted view has no code report")?;
    let n = code.input.n;
    let point = code
        .point
        .as_ref()
        .ok_or("accepted code has no actual point")?;
    if point.len() != 2 * n + code.partition.classes.len() {
        return Err("actual parent point dimension changed".into());
    }
    let request = categorical::Request {
        data: categorical::Data {
            n,
            vocabulary: code.input.vocab,
            observations: code.input.observations.clone(),
        },
        proposal: categorical::Proposal {
            rows: point[..n].to_vec(),
            columns: point[n..2 * n].to_vec(),
            codes: point[2 * n..].to_vec(),
        },
        retained_equations: policy.retained_equations,
        policy: policy.ring.clone(),
    };
    if request.validate()?.0 != code.partition.classes {
        return Err("parent class codes do not follow actual observed class order".into());
    }
    Ok(request)
}

pub(super) fn scope(parent: &search::Report, index: usize) -> Result<Scope> {
    let c = parent
        .candidates
        .get(index)
        .ok_or("missing scope candidate")?;
    let v = &parent
        .attempts
        .get(c.view)
        .ok_or("missing scope view")?
        .view;
    Ok(Scope {
        kind: "fixed_bijective_cyclic_view".into(),
        unknown_bijective_interior_required: true,
        constant_strip_structure_required: v.constant.is_some(),
        constant: v.constant,
        row_map: v.rows.clone(),
        column_map: v.columns.clone(),
        unconditional_ground_truth_identification: false,
    })
}

/// Bind every original model prediction to the actual categorical phase/code
/// coordinates. These are proposed predictions, never held-out target labels.
pub(super) fn predictions(
    parent: &search::Report,
    index: usize,
    r: &categorical::Request,
) -> Result<usize> {
    let c = parent
        .candidates
        .get(index)
        .ok_or("missing model candidate")?;
    c.model.validate()?;
    let (classes, _) = r.validate()?;
    let v = &parent
        .attempts
        .get(c.view)
        .ok_or("missing prediction view")?
        .view;
    if c.model.n != parent.input.categorical.n || c.model.vocab != r.data.vocabulary {
        return Err("original full model shape/vocabulary differs".into());
    }
    let q = classes.len();
    for (j, col) in v.columns.iter().enumerate() {
        for (i, row) in v.rows.iter().enumerate() {
            let expected = match (row, col) {
                (Some(a), Some(b)) => {
                    let z = (r.proposal.rows[*a] as usize + r.proposal.columns[*b] as usize) % q;
                    let code = r
                        .proposal
                        .codes
                        .iter()
                        .position(|&x| x as usize == z)
                        .ok_or("original code lacks decoder state")?;
                    classes[code]
                }
                _ => v.constant.ok_or("original strip has no constant")?,
            };
            if c.model.output_labels[i + c.model.n * j] != expected {
                return Err(
                    "full proposed prediction differs from the certified interior/view".into(),
                );
            }
        }
    }
    Ok(c.model.n * c.model.n)
}

pub(super) fn views(parent: &search::Report) -> Vec<Value> {
    parent
        .attempts
        .iter()
        .enumerate()
        .map(|(i, a)| {
            serde_json::json!({"view":i,
        "decision":a.decision,"coding_reason":a.coding.as_ref().map(|c|&c.reason),
        "row_padding":a.view.row_padding,"column_padding":a.view.column_padding,
        "missing_rows":a.view.missing_rows,"missing_columns":a.view.missing_columns})
        })
        .collect()
}

pub(super) fn forecast(parent: &search::Report, policy: &Policy) -> Result<u64> {
    policy.validate()?;
    let mut bound = parent.input.validate()? + 8 * 1024 * 1024 * 1024;
    for index in 0..parent.candidates.len() {
        bound += request(parent, index, policy)?.validate()?.1;
    }
    if bound > 56 * 1024 * 1024 * 1024 {
        return Err("parent/certificate/JSON forecast exceeds56GiB".into());
    }
    Ok(bound)
}

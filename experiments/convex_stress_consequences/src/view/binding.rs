use super::*;

/// Independent construction of the actual tagged phase table, without calling
/// View::lift or Model::construct. All output names stay in the full vocabulary.
pub(super) fn reconstruct(view: &views::View, base: &Model) -> Result<Model> {
    if view.constant.is_none() {
        return Ok(base.clone());
    }
    let constant = view.constant.ok_or("tag lift lost its constant")?;
    let n = view.rows.len();
    let q = 3 * base.q;
    let rows: Vec<_> = view
        .rows
        .iter()
        .map(|r| r.map_or(1, |r| 3 * base.rows[r]))
        .collect();
    let columns: Vec<_> = view
        .columns
        .iter()
        .map(|c| c.map_or(1, |c| 3 * base.columns[c]))
        .collect();
    let decoder: Vec<_> = (0..q)
        .map(|z| {
            if z % 3 == 0 {
                base.decoder[z / 3]
            } else {
                constant
            }
        })
        .collect();
    let latent_labels: Vec<_> = (0..n * n)
        .map(|cell| (rows[cell % n] + columns[cell / n]) % q)
        .collect();
    let output_labels = latent_labels.iter().map(|&z| decoder[z]).collect();
    Ok(Model {
        n,
        q,
        vocab: base.vocab,
        rows,
        columns,
        decoder,
        latent_labels,
        output_labels,
    })
}
pub(super) fn scope(view: &views::View, base: &learner::Candidate, full: &Model) -> Result<Scope> {
    let cert = &base.certificate;
    if cert.input.predictions != base.model.output_labels
        || cert.input.observations != view.observations
    {
        return Err(
            "view certificate is not bound to its actual interior model and observations".into(),
        );
    }
    let mut out = Scope {
        theorem: "Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.chooseView_sound".into(),
        target_family: "Unknown target uses these fixed maps and strip constant; interior labels lie in class_to_output and have an unknown injective code into the stated residue ring, with additive row/column phases fitting the interior observations.".into(),
        modulus: base.model.q,
        class_to_output: cert.classes.clone(),
        row_map: view.rows.clone(),
        column_map: view.columns.clone(),
        constant: view.constant,
        origins: vec![],
        certified_interior_cells: 0,
        assumed_strip_cells: 0,
        uncertified_cells: 0,
        full_table_certified_within_family: false,
        unconditional_identification: false,
    };
    for (j, column) in view.columns.iter().enumerate() {
        for (i, row) in view.rows.iter().enumerate() {
            let (expected, origin) = if let (Some(r), Some(c)) = (row, column) {
                let cell = r + base.model.n * c;
                let expected = base.model.output_labels[cell];
                let passed = cert.queries.get(cell).is_some_and(|q| {
                    q.cell == cell && q.label == expected && q.decision == decoded::Decision::Passed
                });
                let origin = if passed {
                    out.certified_interior_cells += 1;
                    Origin::ObservedConsequence {
                        interior_cell: cell,
                        query: cell,
                    }
                } else {
                    out.uncertified_cells += 1;
                    Origin::UncertifiedInterior {
                        interior_cell: cell,
                    }
                };
                (expected, origin)
            } else {
                out.assumed_strip_cells += 1;
                (
                    view.constant
                        .ok_or("unmapped full input has no declared constant")?,
                    Origin::AssumedConstantStrip,
                )
            };
            if full.output_labels[i + full.n * j] != expected {
                return Err(
                    "full output differs from the fixed categorical map or strip label".into(),
                );
            }
            out.origins.push(origin);
        }
    }
    out.full_table_certified_within_family = out.uncertified_cells == 0;
    Ok(out)
}

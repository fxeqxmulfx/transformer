use super::*;
pub fn run(request: Request) -> Result<Report> {
    let (names, problem, bytes) = request.problem()?;
    let field = finite_system::solve(problem.clone())?;
    let field_audit = finite_system::check(&problem, &field)?;
    let mut out = Report {
        request,
        buffer_preflight_bytes: bytes,
        names,
        reason: field.reason.clone(),
        work: Work {
            field: field.work.clone(),
            ..Default::default()
        },
        field,
        field_audit,
        anchor_class: None,
        inverse: None,
        code: None,
        conflict: None,
        code_fibers: vec![],
        raw_label_identification: false,
    };
    let Some(space) = &out.field.space else {
        return Ok(out);
    };
    if space.directions.len() != 1 {
        out.reason = "no_nonconstant_direction".into();
        return Ok(out);
    }
    let (nr, nc) = (out.names.rows.len(), out.names.columns.len());
    let direction = &space.directions[0];
    let Some(anchor) = (0..out.names.classes.len()).find(|&c| direction[nr + nc + c] != 0) else {
        out.reason = "input_only_direction".into();
        return Ok(out);
    };
    let prime = out.request.prime;
    let inverse = (1..prime)
        .find(|&x| {
            out.work.inverse_tests += 1;
            (x as u32 * direction[nr + nc + anchor] as u32) % prime as u32 == 1
        })
        .ok_or("nonzero field coordinate lacks an inverse")?;
    let point: Vec<_> = direction
        .iter()
        .map(|&x| ((x as u32 * inverse as u32) % prime as u32) as u16)
        .collect();
    out.work.normalization_products = point.len();
    out.anchor_class = Some(anchor);
    out.inverse = Some(inverse);
    let code = Code {
        prime,
        names: out.names.clone(),
        rows: point[..nr].into(),
        columns: point[nr..nr + nc].into(),
        classes: point[nr + nc..].into(),
    };
    for (index, &[cell, label]) in out.request.input.observations.iter().enumerate() {
        out.work.observation_checks += 1;
        if code.query(cell % out.request.input.n, cell / out.request.input.n) != code.label(label) {
            out.conflict = Some(index);
            out.reason = "original_observation_conflict".into();
            return Ok(out);
        }
    }
    out.code_fibers = plan::fibers(&code);
    out.code = Some(code);
    out.reason = "admitted_nonconstant_code_with_possible_aliases".into();
    Ok(out)
}

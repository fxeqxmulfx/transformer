//! Independent completeness check: replay the original finite row system,
//! then test all original equations and full rank on every free coordinate.
//! RREF establishes the kernel dimension; an invertible free-coordinate matrix
//! proves that an alternate basis spans it, rather than merely fitting rows.
use super::*;
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BasisAudit {
    pub rank: usize,
    pub dimension: usize,
    pub original_equation_products: u64,
    pub free_matrix_products: u64,
    pub complete: bool,
}
fn inverse(a: i64, p: i64) -> Result<i64> {
    let (mut r, mut next, mut t, mut next_t) = (p, a, 0, 1);
    while next != 0 {
        let quotient = r / next;
        (r, next) = (next, r - quotient * next);
        (t, next_t) = (next_t, t - quotient * next_t);
    }
    if r != 1 {
        return Err("free-coordinate pivot is not a field unit".into());
    }
    Ok(t.rem_euclid(p))
}
pub(crate) fn basis_after_field_check(
    problem: &finite_system::Problem,
    field: &finite_system::Outcome,
    basis: &[Vec<u16>],
) -> Result<BasisAudit> {
    let s = &problem.system;
    let dimension = field.free.len();
    if field.reason != "complete"
        || field.space.is_none()
        || s.right.iter().any(|&x| x != 0)
        || dimension > 8
        || basis.len() != dimension
        || basis
            .iter()
            .any(|v| v.len() != s.variables || v.iter().any(|&x| x >= s.prime))
    {
        return Err("basis is not complete for the original homogeneous kernel".into());
    }
    let mut products = 0;
    for direction in basis {
        for row in &s.rows {
            let value: i128 = row
                .iter()
                .zip(direction)
                .map(|(&a, &x)| {
                    products += 1;
                    i128::from(a) * i128::from(x)
                })
                .sum();
            if value.rem_euclid(i128::from(s.prime)) != 0 {
                return Err(
                    "basis direction fails an original equation or legitimate gauge".into(),
                );
            }
        }
    }
    let p = i64::from(s.prime);
    let mut free: Vec<Vec<i64>> = basis
        .iter()
        .map(|v| field.free.iter().map(|&k| i64::from(v[k])).collect())
        .collect();
    let mut free_products = 0;
    for k in 0..dimension {
        let row = (k..dimension)
            .find(|&i| free[i][k] != 0)
            .ok_or("basis free-coordinate matrix is singular")?;
        free.swap(k, row);
        let inv = inverse(free[k][k], p)?;
        for value in &mut free[k][k..] {
            *value = (*value * inv).rem_euclid(p);
            free_products += 1;
        }
        let pivot = free[k][k..].to_vec();
        for row in free.iter_mut().skip(k + 1) {
            let factor = row[k];
            for (value, &source) in row[k..].iter_mut().zip(&pivot) {
                *value = (*value - factor * source).rem_euclid(p);
                free_products += 1;
            }
        }
    }
    Ok(BasisAudit {
        rank: field.pivots.len(),
        dimension,
        original_equation_products: products,
        free_matrix_products: free_products,
        complete: true,
    })
}
pub fn check_basis(
    problem: &finite_system::Problem,
    field: &finite_system::Outcome,
    basis: &[Vec<u16>],
) -> Result<BasisAudit> {
    finite_system::check(problem, field)?;
    basis_after_field_check(problem, field, basis)
}
pub fn check(request: &Request, out: &Report) -> Result<()> {
    let names = request.names()?;
    if &out.request != request
        || out.names != names
        || out.raw_label_identification
        || out.target_readout_compatibility_proved
    {
        return Err("kernel input, categorical names or conditional scope changed".into());
    }
    if let Some(reason) = request.refusal(&names) {
        if out.reason != reason
            || out.buffer_preflight_bytes != 0
            || out.field.is_some()
            || out.field_audit.is_some()
            || out.basis_audit.is_some()
            || out.code.is_some()
            || out.work != Work::default()
        {
            return Err("pre-allocation kernel refusal was altered".into());
        }
        return Ok(());
    }
    let problem = request.problem(&names)?;
    let bytes = 4 * problem.validate()? + 512 * 1024 * 1024;
    if out.buffer_preflight_bytes != bytes {
        return Err("kernel memory forecast changed".into());
    }
    if bytes > 8 * 1024 * 1024 * 1024 {
        if out.reason != "discovery_memory_cap"
            || out.field.is_some()
            || out.field_audit.is_some()
            || out.basis_audit.is_some()
            || out.code.is_some()
            || out.work != Work::default()
        {
            return Err("kernel memory refusal changed".into());
        }
        return Ok(());
    }
    let field = out.field.as_ref().ok_or("missing original field outcome")?;
    let audit = finite_system::check(&problem, field)?;
    if out.field_audit.as_ref() != Some(&audit)
        || out.work.field != field.work
        || out.work.ce_gradients != 0
    {
        return Err("original field audit or discovery work changed".into());
    }
    match &field.space {
        None => {
            if out.reason != field.reason
                || out.basis_audit.is_some()
                || out.code.is_some()
                || out.work.code_coordinates != 0
                || out.work.observed_vector_checks != 0
            {
                return Err("partial field outcome was promoted to a complete code".into());
            }
        }
        Some(space) => {
            let basis = basis_after_field_check(&problem, field, &space.directions)?;
            let code = Code::from_directions(names, request.prime, &space.directions)?;
            code.validate(request.input.n, request.input.vocabulary)?;
            if out.reason != "complete_kernel_vector_code"
                || out.basis_audit.as_ref() != Some(&basis)
                || out.code.as_ref() != Some(&code)
                || out.work.code_coordinates != space.directions.len() * problem.system.variables
                || out.work.observed_vector_checks != request.input.observations.len()
            {
                return Err("complete vector code, basis or costs changed".into());
            }
            for &[cell, label] in &request.input.observations {
                if code.query(cell % request.input.n, cell / request.input.n)
                    != code.label(label).map(<[u16]>::to_vec)
                {
                    return Err("full-vector observed-label fit failed".into());
                }
            }
        }
    }
    Ok(())
}

use super::*;
pub fn check(input: &Request, out: &Report) -> Result<()> {
    let bound = input.validate()?;
    if out.input != *input
        || out.buffer_preflight_bytes != bound
        || out.points.len() > input.policy.points
        || out.elimination_audit != finite_system::check(&input.problem, &out.elimination)?
    {
        return Err("affine original field certificate, policy or extent changed".into());
    }
    let Some(space) = &out.elimination.space else {
        if !out.points.is_empty()
            || out.cardinality.is_some()
            || out.cardinality_overflow
            || out.complete
            || out.work != Work::default()
            || out.reason != format!("elimination_{}", out.elimination.reason)
        {
            return Err("unfinished/refused field space supplied affine points".into());
        }
        return Ok(());
    };
    let s = &input.problem.system;
    let q = s.prime as u128;
    let d = space.directions.len();
    let exact = q.pow(d as u32);
    let cardinality = u64::try_from(exact).ok();
    if out.cardinality != cardinality
        || out.cardinality_overflow != cardinality.is_none()
        || out.points.len() as u128 > exact
    {
        return Err("actual affine-space cardinality changed".into());
    }
    for (ordinal, point) in out.points.iter().enumerate() {
        if point.ordinal != ordinal
            || point.weights.len() != d
            || point.values.len() != s.variables
            || point.values.iter().any(|&v| v >= s.prime)
        {
            return Err("affine point dimension, residue or original ordering changed".into());
        }
        // Independent digit extraction and one unreduced signed dot product.
        for (k, &weight) in point.weights.iter().enumerate() {
            if weight as u128 != (ordinal as u128 / q.pow(k as u32)) % q {
                return Err("affine parameter tuple changed".into());
            }
        }
        for (j, &value) in point.values.iter().enumerate() {
            let actual = space.particular[j] as i128
                + space
                    .directions
                    .iter()
                    .zip(&point.weights)
                    .map(|(direction, &weight)| direction[j] as i128 * weight as i128)
                    .sum::<i128>();
            if actual.rem_euclid(s.prime as i128) != value as i128 {
                return Err("affine point is not the stated actual combination".into());
            }
        }
        for (row, &right) in s.rows.iter().zip(&s.right) {
            let actual: i128 = row
                .iter()
                .zip(&point.values)
                .map(|(&a, &x)| a as i128 * x as i128)
                .sum();
            if actual.rem_euclid(s.prime as i128) != right as i128 {
                return Err("affine point fails original-row i128 replay".into());
            }
        }
    }
    let count = out.points.len();
    let step_coordinates = (d * s.variables) as u64;
    let step_original = (s.rows.len() * s.variables) as u64;
    let work = Work {
        points: count,
        parameter_digits: count * d,
        coordinate_products: count as u64 * step_coordinates,
        original_products: count as u64 * step_original,
        ce_gradients: 0,
    };
    if out.work != work
        || work.coordinate_products > input.policy.coordinate_products
        || work.original_products > input.policy.original_products
    {
        return Err("affine exact work or cap differs".into());
    }
    let (complete, reason) = if count as u128 == exact {
        (true, "enumerated_entire_space")
    } else if count == input.policy.points {
        (false, "point_cap")
    } else if work.coordinate_products + step_coordinates > input.policy.coordinate_products {
        (false, "coordinate_product_cap")
    } else if work.original_products + step_original > input.policy.original_products {
        (false, "original_product_cap")
    } else {
        return Err("affine producer stopped before any declared limit".into());
    };
    if out.complete != complete || out.reason != reason {
        return Err("affine completion or stop reason differs".into());
    }
    Ok(())
}

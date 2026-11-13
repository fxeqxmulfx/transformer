use super::*;
pub fn run(input: Request) -> Result<Report> {
    let bound = input.validate()?;
    let elimination = finite_system::solve(input.problem.clone())?;
    let audit = finite_system::check(&input.problem, &elimination)?;
    let mut out = Report {
        input,
        buffer_preflight_bytes: bound,
        elimination,
        elimination_audit: audit,
        cardinality: None,
        cardinality_overflow: false,
        points: vec![],
        complete: false,
        reason: String::new(),
        work: Work::default(),
    };
    let Some(space) = &out.elimination.space else {
        out.reason = format!("elimination_{}", out.elimination.reason);
        return Ok(out);
    };
    let s = &out.input.problem.system;
    let p = &out.input.policy;
    let d = space.directions.len();
    out.cardinality = (s.prime as u64).checked_pow(d as u32);
    out.cardinality_overflow = out.cardinality.is_none();
    let next_coordinates = (d * s.variables) as u64;
    let next_original = (s.rows.len() * s.variables) as u64;
    loop {
        if out.cardinality == Some(out.points.len() as u64) {
            out.complete = true;
            out.reason = "enumerated_entire_space".into();
            break;
        }
        if out.points.len() == p.points {
            out.reason = "point_cap".into();
            break;
        }
        if out.work.coordinate_products + next_coordinates > p.coordinate_products {
            out.reason = "coordinate_product_cap".into();
            break;
        }
        if out.work.original_products + next_original > p.original_products {
            out.reason = "original_product_cap".into();
            break;
        }
        let ordinal = out.points.len();
        let mut index = ordinal;
        let weights: Vec<_> = (0..d)
            .map(|_| {
                let digit = (index % s.prime as usize) as u16;
                index /= s.prime as usize;
                digit
            })
            .collect();
        let mut values = space.particular.clone();
        for (direction, &weight) in space.directions.iter().zip(&weights) {
            for (value, &coefficient) in values.iter_mut().zip(direction) {
                *value =
                    ((*value as u64 + coefficient as u64 * weight as u64) % s.prime as u64) as u16;
            }
        }
        for (row, &right) in s.rows.iter().zip(&s.right) {
            let actual: u64 = row
                .iter()
                .zip(&values)
                .map(|(&a, &x)| a as u64 * x as u64)
                .sum();
            if actual % s.prime as u64 != right as u64 {
                return Err("affine point violates an original equation".into());
            }
        }
        out.points.push(Point {
            ordinal,
            weights,
            values,
        });
        out.work.points += 1;
        out.work.parameter_digits += d;
        out.work.coordinate_products += next_coordinates;
        out.work.original_products += next_original;
    }
    Ok(out)
}

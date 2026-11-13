use super::*;

pub fn check(input: &Request, out: &Report) -> Result<()> {
    let bound = input.validate()?;
    let factors = plan::factors(input.classes);
    if out.input != *input
        || out.buffer_preflight_bytes != bound
        || out.components.len() != factors.len()
        || out.branches.len() > input.policy.points
    {
        return Err("composite original input, factors, bound or prefix differs".into());
    }
    let mut cardinality = 1_u64;
    let mut work = Work::default();
    for (factor, c) in factors.iter().zip(&out.components) {
        component::check(&input.component_request(factor.clone()), c)?;
        cardinality *= c.complete.len() as u64;
        work.components.push(c.work.clone());
    }
    let per_branch = input.products_per_branch(factors.len());
    let extent = cardinality
        .min(input.policy.points as u64)
        .min(input.policy.original_products / per_branch) as usize;
    if out.cardinality != cardinality || out.branches.len() != extent {
        return Err(
            "composite actual emitted-product cardinality or bounded extent differs".into(),
        );
    }
    let mut candidates = vec![];
    for (ordinal, b) in out.branches.iter().enumerate() {
        let mut stride = 1_u64;
        let states: Vec<_> = out
            .components
            .iter()
            .map(|c| {
                let size = c.complete.len() as u64;
                let digit = (ordinal as u64 / stride) % size;
                stride *= size;
                c.complete[digit as usize]
            })
            .collect();
        if b.ordinal != ordinal || b.component_states != states || b.crt.len() != factors.len() - 1
        {
            return Err("composite tuple order or CRT extent differs".into());
        }
        work.tuples += 1;
        work.parameter_digits += factors.len();
        let mut previous = out.components[0].states[states[0]].point.clone();
        let mut modulus = factors[0].modulus;
        for (k, step) in b.crt.iter().enumerate() {
            let factor = &factors[k + 1];
            let request = recombine::Input {
                system: input.system.clone(),
                left_modulus: modulus,
                right_modulus: factor.modulus,
                left: previous,
                right: out.components[k + 1].states[states[k + 1]].point.clone(),
            };
            recombine::check(&request, step)?;
            previous = step.point.clone().ok_or("missing checked CRT point")?;
            modulus *= factor.modulus;
            work.crt_steps += 1;
            work.inverse_tests += step.inverse_tests;
            work.crt_original_products += step.original_products as u64;
        }
        if b.point != previous || modulus as usize != input.classes {
            return Err("composite endpoint is not its actual CRT representative".into());
        }
        input.system.validate_point(modulus, &b.point)?;
        for (row, &right) in input.system.rows.iter().zip(&input.system.right) {
            let value = row
                .iter()
                .zip(&b.point)
                .fold(right as i128, |sum, (&a, &x)| sum - a as i128 * x as i128);
            if value.rem_euclid(modulus as i128) != 0 {
                return Err("composite endpoint violates an original integer equation".into());
            }
        }
        work.endpoint_original_products +=
            (input.system.variables * input.system.rows.len()) as u64;
        // Independently scan class pairs rather than reuse the producer's map.
        let codes = &b.point[2 * input.n..];
        let mut collision = None;
        for (j, &value) in codes.iter().enumerate() {
            work.class_coordinates += 1;
            if let Some(i) = codes[..j].iter().position(|&v| v == value) {
                collision = Some([i, j, value as usize]);
                break;
            }
        }
        let mut duplicate = None;
        if collision.is_none() {
            for &earlier in &candidates {
                work.duplicate_tests += 1;
                let other: &Branch = &out.branches[earlier];
                if other.point.iter().zip(&b.point).all(|(x, y)| x == y) {
                    duplicate = Some(earlier);
                    break;
                }
            }
        }
        let retained = collision.is_none() && duplicate.is_none();
        if b.collision != collision || b.duplicate != duplicate || b.retained != retained {
            return Err("composite actual class collision, duplicate or retention differs".into());
        }
        if retained {
            candidates.push(ordinal);
        }
    }
    let entire = extent as u64 == cardinality;
    let reason = if cardinality == 0 {
        "component_refused"
    } else if entire {
        "enumerated_component_product"
    } else if extent == input.policy.points {
        "point_cap"
    } else {
        "original_product_cap"
    };
    if out.candidates != candidates
        || out.entire_emitted_product != entire
        || out.reason != reason
        || out.work != work
    {
        return Err("composite candidates, coverage boundary, stop or actual work differs".into());
    }
    Ok(())
}

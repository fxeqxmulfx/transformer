use super::*;
pub fn check(input: &Request, out: &Report) -> Result<()> {
    input.validate()?;
    let m = input.modulus as i128;
    let p = input.prime as i128;
    let mut residual = vec![];
    for (row, &right) in input.system.rows.iter().zip(&input.system.right) {
        let value = right as i128
            - row
                .iter()
                .zip(&input.base)
                .map(|(&a, &x)| a as i128 * x as i128)
                .sum::<i128>();
        if value.rem_euclid(m) != 0 {
            return Err("branch base fails original i128 congruence".into());
        }
        residual.push(i64::try_from(value / m)?);
    }
    let field = affine::Request {
        problem: finite_system::Problem {
            system: finite_system::System {
                prime: input.prime,
                variables: input.system.variables,
                rows: input
                    .system
                    .rows
                    .iter()
                    .map(|row| {
                        row.iter()
                            .map(|&a| ((a as i128 % p + p) % p) as u16)
                            .collect()
                    })
                    .collect(),
                right: residual
                    .iter()
                    .map(|&b| ((b as i128 % p + p) % p) as u16)
                    .collect(),
            },
            policy: input.elimination.clone(),
        },
        policy: input.enumeration.clone(),
    };
    if out.input != *input
        || out.residual != residual
        || out.buffer_preflight_bytes != input.bound(&field)?
        || out.branches.len() != out.digits.points.len()
        || out.branches.len() > input.enumeration.points
        || out.entire_digit_space != out.digits.complete
        || out.reason != format!("digits_{}", out.digits.reason)
    {
        return Err("branched lift original input/residual/bound/coverage differs".into());
    }
    affine::check(&field, &out.digits)?;
    for (branch, digit) in out.branches.iter().zip(&out.digits.points) {
        if branch.point.len() != input.system.variables
            || branch.original_difference.len() != input.system.rows.len()
        {
            return Err("lifted branch shape differs".into());
        }
        for ((&x, &y), &z) in input.base.iter().zip(&digit.values).zip(&branch.point) {
            if z as i128 != x as i128 + m * y as i128 || z as i128 >= m * p {
                return Err("actual branch digit identity fails".into());
            }
        }
        for (index, (row, &right)) in input
            .system
            .rows
            .iter()
            .zip(&input.system.right)
            .enumerate()
        {
            let value = right as i128
                - row
                    .iter()
                    .zip(&branch.point)
                    .map(|(&a, &x)| a as i128 * x as i128)
                    .sum::<i128>();
            let digit_action: i128 = row
                .iter()
                .zip(&digit.values)
                .map(|(&a, &y)| a as i128 * y as i128)
                .sum();
            if value.rem_euclid(m * p) != 0
                || value != branch.original_difference[index] as i128
                || value != m * (residual[index] as i128 - digit_action)
            {
                return Err("original branch or exact CM27 residual identity fails".into());
            }
        }
    }
    let products = (input.system.variables * input.system.rows.len()) as u64;
    if out.work
        != (Work {
            base_products: products,
            lifted_products: out.branches.len() as u64 * products,
            ce_gradients: 0,
        })
    {
        return Err("branch original arithmetic work differs".into());
    }
    Ok(())
}

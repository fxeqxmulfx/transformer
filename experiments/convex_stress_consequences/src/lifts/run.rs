use super::*;
pub fn run(input: Request) -> Result<Report> {
    input.validate()?;
    let mut residual = vec![];
    for (row, &b) in input.system.rows.iter().zip(&input.system.right) {
        let difference = b - row
            .iter()
            .zip(&input.base)
            .map(|(&a, &x)| a * x as i64)
            .sum::<i64>();
        if difference % (input.modulus as i64) != 0 {
            return Err("branch base fails an original congruence".into());
        }
        residual.push(difference / input.modulus as i64);
    }
    let p = input.prime as i64;
    let field = affine::Request {
        problem: finite_system::Problem {
            system: finite_system::System {
                prime: input.prime,
                variables: input.system.variables,
                rows: input
                    .system
                    .rows
                    .iter()
                    .map(|row| row.iter().map(|a| a.rem_euclid(p) as u16).collect())
                    .collect(),
                right: residual.iter().map(|b| b.rem_euclid(p) as u16).collect(),
            },
            policy: input.elimination.clone(),
        },
        policy: input.enumeration.clone(),
    };
    let bound = input.bound(&field)?;
    let digits = affine::run(field)?;
    let mut branches = vec![];
    for digit in &digits.points {
        let point: Vec<_> = input
            .base
            .iter()
            .zip(&digit.values)
            .map(|(&x, &y)| x + input.modulus * y)
            .collect();
        let original_difference: Vec<_> = input
            .system
            .rows
            .iter()
            .zip(&input.system.right)
            .map(|(row, &b)| {
                b - row
                    .iter()
                    .zip(&point)
                    .map(|(&a, &x)| a * x as i64)
                    .sum::<i64>()
            })
            .collect();
        if original_difference
            .iter()
            .any(|d| d % (input.modulus as i64 * input.prime as i64) != 0)
        {
            return Err("branch violates an original next-modulus equation".into());
        }
        branches.push(Branch {
            point,
            original_difference,
        });
    }
    let products = (input.system.rows.len() * input.system.variables) as u64;
    let work = Work {
        base_products: products,
        lifted_products: branches.len() as u64 * products,
        ce_gradients: 0,
    };
    Ok(Report {
        input,
        residual,
        entire_digit_space: digits.complete,
        reason: format!("digits_{}", digits.reason),
        digits,
        branches,
        buffer_preflight_bytes: bound,
        work,
    })
}

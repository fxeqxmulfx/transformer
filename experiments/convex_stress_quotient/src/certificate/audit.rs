use super::*;

fn row(input: &Request, variables: usize, cell: usize, class: usize) -> Vec<i128> {
    (0..variables)
        .map(|v| {
            (i128::from(v == cell % input.n) + i128::from(v == input.n + cell / input.n)
                - i128::from(v == 2 * input.n + class))
            .rem_euclid(input.modulus as i128)
        })
        .collect()
}

// Recompute query reductions in signed i128, then replay the actual suffix.
// It changes only its receiver, so every source is the checked prefix row.
fn query(
    input: &Request,
    classes: &[usize],
    prefix: &row_program::Outcome,
    work: &mut Work,
    cell: usize,
) -> Result<Query> {
    work.query_rows += 1;
    let label = input.predictions[cell];
    let class = classes.iter().position(|&c| c == label);
    let mut expected = Query {
        cell,
        label,
        class,
        terms: vec![],
        residual: vec![],
        receiver: None,
        operations: vec![],
        decision: Decision::UnknownClass,
    };
    let Some(class) = class else {
        return Ok(expected);
    };
    let variables = prefix.input.variables;
    let target = row(input, variables, cell, class);
    let mut residue = target.clone();
    let q = input.modulus as i128;
    let mut failure = None;
    for pivot in &prefix.pivots {
        work.pivot_tests += 1;
        let value = residue[pivot.column];
        if value == 0 {
            continue;
        }
        if value % pivot.divisor as i128 != 0 {
            failure = Some(Decision::Nondivisible {
                column: pivot.column,
                value: value as u16,
                divisor: pivot.divisor,
            });
            break;
        }
        if work.reduction_products + variables as u64 > input.policy.reduction_products {
            failure = Some(Decision::ReductionCap);
            break;
        }
        let factor = value / pivot.divisor as i128;
        residue = residue
            .iter()
            .zip(&prefix.matrix[pivot.row])
            .map(|(&x, &a)| (x - factor * a as i128).rem_euclid(q))
            .collect();
        expected.terms.push(Term {
            row: pivot.row,
            factor: factor as u16,
        });
        work.reduction_products += variables as u64;
    }
    expected.residual = residue.iter().map(|&x| x as u16).collect();
    if let Some(failure) = failure {
        expected.decision = failure;
        return Ok(expected);
    }
    if residue.iter().any(|&x| x != 0) {
        expected.decision = Decision::Unrepresented;
        return Ok(expected);
    }
    let first = expected
        .terms
        .first()
        .ok_or("zero terms cannot represent a categorical query")?;
    let receiver = first.row;
    let mut operations = vec![];
    if first.factor != 1 {
        operations.push(Op::Scale {
            i: receiver,
            factor: first.factor,
        });
    }
    operations.extend(expected.terms.iter().skip(1).map(|t| Op::Add {
        i: receiver,
        j: t.row,
        factor: t.factor,
    }));
    if work.certificate_operations + operations.len() > input.policy.certificate_operations {
        expected.decision = Decision::CertificateOperationCap;
        return Ok(expected);
    }
    let products = operations.len() as u64 * variables as u64;
    if work.certificate_products + products > input.policy.certificate_products {
        expected.decision = Decision::CertificateProductCap;
        return Ok(expected);
    }
    let mut actual: Vec<i128> = prefix.matrix[receiver].iter().map(|&x| x as i128).collect();
    for op in &operations {
        match *op {
            Op::Scale { i, factor } if i == receiver && factor > 0 && factor < input.modulus => {
                actual = actual
                    .iter()
                    .map(|x| (factor as i128 * x).rem_euclid(q))
                    .collect();
            }
            Op::Add { i, j, factor }
                if i == receiver
                    && j != receiver
                    && j < prefix.matrix.len()
                    && factor > 0
                    && factor < input.modulus =>
            {
                actual = actual
                    .iter()
                    .zip(&prefix.matrix[j])
                    .map(|(&x, &a)| (x + factor as i128 * a as i128).rem_euclid(q))
                    .collect();
            }
            _ => return Err("query suffix is not an actual fixed-receiver row program".into()),
        }
    }
    work.coefficient_checks += variables as u64;
    if actual != target {
        return Err(
            "query certificate fails its actual coefficient-by-coefficient identity".into(),
        );
    }
    work.certificate_operations += operations.len();
    work.certificate_products += products;
    expected.receiver = Some(receiver);
    expected.operations = operations;
    expected.decision = Decision::Passed;
    Ok(expected)
}

pub fn check(input: &Request, out: &Report) -> Result<()> {
    let (classes, bound) = input.validate()?;
    if out.input != *input
        || out.classes != classes
        || out.buffer_preflight_bytes != bound
        || out.queries.len() > input.predictions.len().min(input.policy.queries)
    {
        return Err("decoded original observations, class family, shape or bound differs".into());
    }
    let variables = 2 * input.n + classes.len();
    // Reconstruct from the observations themselves. There are no gauge,
    // normalization, missing-axis or proposed-answer rows in this matrix.
    let rows = input
        .observations
        .iter()
        .take(input.policy.equations)
        .map(|&[cell, label]| {
            let class = classes
                .iter()
                .position(|&c| c == label)
                .expect("observed class");
            row(input, variables, cell, class)
                .iter()
                .map(|&x| x as u16)
                .collect()
        })
        .collect();
    row_program::check(
        &row_program::Problem {
            modulus: input.modulus,
            variables,
            rows,
            policy: input.prefix_policy.clone(),
        },
        &out.prefix,
    )?;
    let mut work = Work {
        prefix: out.prefix.work.clone(),
        ..Work::default()
    };
    let mut passed = 0;
    let mut capped = false;
    let mut length = 0;
    for cell in 0..input.predictions.len().min(input.policy.queries) {
        let expected = query(input, &classes, &out.prefix, &mut work, cell)?;
        if out.queries.get(cell) != Some(&expected) {
            return Err(
                "decoded original query, divisor reduction, actual program or cap differs".into(),
            );
        }
        length += 1;
        passed += usize::from(expected.decision == Decision::Passed);
        if expected.decision.capped() {
            capped = true;
            break;
        }
    }
    let certified = passed == input.predictions.len();
    let reason = if certified {
        "code_equal_under_additive_realizability_quotient_requires_compatibility"
    } else if capped {
        "query_work_cap"
    } else if length < input.predictions.len() {
        "query_count_cap"
    } else {
        "uncertified_queries"
    };
    if out.queries.len() != length
        || out.raw_label_identification
        || out.certified != certified
        || out.reason != reason
        || out.work != work
    {
        return Err(
            "decoded coverage, conditional certificate status, termination or work differs".into(),
        );
    }
    Ok(())
}

use super::*;

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
    let mut out = Query {
        cell,
        label,
        class,
        terms: vec![],
        residual: vec![],
        receiver: None,
        operations: vec![],
        decision: Decision::UnknownClass,
    };
    let Some(class) = class else { return Ok(out) };
    let target = input.row(classes.len(), cell, class);
    out.residual = target.clone();
    let variables = target.len() as u64;
    let q = input.modulus as u32;
    for pivot in &prefix.pivots {
        work.pivot_tests += 1;
        let value = out.residual[pivot.column];
        if value == 0 {
            continue;
        }
        if !value.is_multiple_of(pivot.divisor) {
            out.decision = Decision::Nondivisible {
                column: pivot.column,
                value,
                divisor: pivot.divisor,
            };
            return Ok(out);
        }
        if variables > input.policy.reduction_products - work.reduction_products {
            out.decision = Decision::ReductionCap;
            return Ok(out);
        }
        let factor = value / pivot.divisor;
        for (x, &b) in out.residual.iter_mut().zip(&prefix.matrix[pivot.row]) {
            *x = ((*x as u32 + q - factor as u32 * b as u32 % q) % q) as u16;
        }
        out.terms.push(Term {
            row: pivot.row,
            factor,
        });
        work.reduction_products += variables;
    }
    if out.residual.iter().any(|&x| x != 0) {
        out.decision = Decision::Unrepresented;
        return Ok(out);
    }
    if out.terms.is_empty() {
        return Err("nonzero categorical query has an empty zero-residual representation".into());
    }
    let (receiver, operations) = plan::suffix(&out.terms);
    if operations.len() > input.policy.certificate_operations - work.certificate_operations {
        out.decision = Decision::CertificateOperationCap;
        return Ok(out);
    }
    let products = operations.len() as u64 * variables;
    if products > input.policy.certificate_products - work.certificate_products {
        out.decision = Decision::CertificateProductCap;
        return Ok(out);
    }
    let mut actual = prefix.matrix[receiver].clone();
    for op in &operations {
        match *op {
            Op::Scale { i, factor } if i == receiver => {
                for x in &mut actual {
                    *x = (*x as u32 * factor as u32 % q) as u16;
                }
            }
            Op::Add { i, j, factor } if i == receiver && j != receiver => {
                for (x, &b) in actual.iter_mut().zip(&prefix.matrix[j]) {
                    *x = ((*x as u32 + factor as u32 * b as u32) % q) as u16;
                }
            }
            _ => return Err("constructed query suffix changed a source row".into()),
        }
    }
    for (actual, expected) in actual.iter().zip(&target) {
        work.coefficient_checks += 1;
        if actual != expected {
            return Err(
                "constructed query program does not produce its actual coefficient row".into(),
            );
        }
    }
    work.certificate_products += products;
    work.certificate_operations += operations.len();
    out.receiver = Some(receiver);
    out.operations = operations;
    out.decision = Decision::Passed;
    Ok(out)
}
pub fn run(input: Request) -> Result<Report> {
    let (classes, bound) = input.validate()?;
    let prefix = row_program::run(input.problem(&classes))?;
    let mut work = Work {
        prefix: prefix.work.clone(),
        ..Work::default()
    };
    let mut queries = vec![];
    for cell in 0..input.predictions.len().min(input.policy.queries) {
        let result = query(&input, &classes, &prefix, &mut work, cell)?;
        let capped = result.decision.capped();
        queries.push(result);
        if capped {
            break;
        }
    }
    let certified = queries.len() == input.predictions.len()
        && queries.iter().all(|q| q.decision == Decision::Passed);
    let reason = if certified {
        "code_equal_under_additive_realizability_quotient_requires_compatibility"
    } else if queries.last().is_some_and(|q| q.decision.capped()) {
        "query_work_cap"
    } else if queries.len() < input.predictions.len() {
        "query_count_cap"
    } else {
        "uncertified_queries"
    }
    .into();
    Ok(Report {
        input,
        buffer_preflight_bytes: bound,
        classes,
        prefix,
        queries,
        certified,
        raw_label_identification: false,
        reason,
        work,
    })
}

use super::*;

pub fn run(input: Request) -> Result<Report> {
    let bound = input.validate()?;
    let mut out = Report {
        input,
        buffer_preflight_bytes: bound,
        components: vec![],
        cardinality: 1,
        branches: vec![],
        candidates: vec![],
        entire_emitted_product: false,
        reason: String::new(),
        work: Work::default(),
    };
    for factor in plan::factors(out.input.classes) {
        let report = component::run(out.input.component_request(factor))?;
        out.cardinality *= report.complete.len() as u64;
        out.work.components.push(report.work.clone());
        out.components.push(report);
    }
    if out.cardinality == 0 {
        out.reason = "component_refused".into();
        out.entire_emitted_product = true;
        return Ok(out);
    }
    let cost = out.input.products_per_branch(out.components.len());
    for ordinal in 0..out.input.policy.points {
        if ordinal as u64 == out.cardinality {
            break;
        }
        let used = out.work.crt_original_products + out.work.endpoint_original_products;
        if cost > out.input.policy.original_products - used {
            break;
        }
        let mut remaining = ordinal;
        let mut states = vec![];
        for c in &out.components {
            states.push(c.complete[remaining % c.complete.len()]);
            remaining /= c.complete.len();
            out.work.parameter_digits += 1;
        }
        let mut point = out.components[0].states[states[0]].point.clone();
        let mut modulus = out.components[0].input.factor.modulus;
        let mut crt = vec![];
        for (c, &state) in out.components.iter().zip(&states).skip(1) {
            let step = recombine::run(recombine::Input {
                system: out.input.system.clone(),
                left_modulus: modulus,
                right_modulus: c.input.factor.modulus,
                left: point,
                right: c.states[state].point.clone(),
            })?;
            out.work.crt_steps += 1;
            out.work.inverse_tests += step.inverse_tests;
            out.work.crt_original_products += step.original_products as u64;
            point = step.point.clone().ok_or("coprime component CRT refused")?;
            modulus *= c.input.factor.modulus;
            crt.push(step);
        }
        if modulus as usize != out.input.classes {
            return Err("CRT product differs from actual class count".into());
        }
        out.input.system.validate_point(modulus, &point)?;
        for (row, &right) in out.input.system.rows.iter().zip(&out.input.system.right) {
            let value: i64 = row.iter().zip(&point).map(|(&a, &x)| a * x as i64).sum();
            if (right - value) % modulus as i64 != 0 {
                return Err("composite endpoint fails its original equations".into());
            }
        }
        out.work.endpoint_original_products +=
            (out.input.system.variables * out.input.system.rows.len()) as u64;
        let mut seen = vec![None; out.input.classes];
        let mut collision = None;
        for (class, &value) in point[2 * out.input.n..].iter().enumerate() {
            out.work.class_coordinates += 1;
            if let Some(earlier) = seen[value as usize] {
                collision = Some([earlier, class, value as usize]);
                break;
            }
            seen[value as usize] = Some(class);
        }
        let duplicate = if collision.is_none() {
            out.branches.iter().enumerate().find_map(|(index, b)| {
                if b.retained {
                    out.work.duplicate_tests += 1;
                    (b.point == point).then_some(index)
                } else {
                    None
                }
            })
        } else {
            None
        };
        let retained = collision.is_none() && duplicate.is_none();
        if retained {
            out.candidates.push(ordinal);
        }
        out.branches.push(Branch {
            ordinal,
            component_states: states,
            crt,
            point,
            collision,
            duplicate,
            retained,
        });
        out.work.tuples += 1;
    }
    out.entire_emitted_product = out.branches.len() as u64 == out.cardinality;
    out.reason = if out.entire_emitted_product {
        "enumerated_component_product"
    } else if out.branches.len() == out.input.policy.points {
        "point_cap"
    } else {
        "original_product_cap"
    }
    .into();
    Ok(out)
}

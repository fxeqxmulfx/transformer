use super::*;

pub(super) fn run(
    system: &System,
    n: usize,
    classes: usize,
    prime: u16,
    policy: finite_system::Policy,
) -> Result<Base> {
    system.validate()?;
    if system.variables != 2 * n + classes || system.right.iter().any(|&b| b != 0) {
        return Err("base coding system is not the declared homogeneous matrix".into());
    }
    let problem = plan::field(system, prime, policy);
    let field = finite_system::solve(problem.clone())?;
    let audit = finite_system::check(&problem, &field)?;
    let mut out = Base {
        reason: field.reason.clone(),
        prime,
        field,
        audit,
        anchor_class: None,
        inverse: None,
        inverse_tests: 0,
        point: None,
    };
    let Some(space) = &out.field.space else {
        return Ok(out);
    };
    if space.directions.len() != 1 {
        out.reason = "zero_kernel".into();
        return Ok(out);
    }
    let direction = &space.directions[0];
    let Some(anchor) = (0..classes).find(|&i| direction[2 * n + i] != 0) else {
        out.reason = "class_constant_direction".into();
        return Ok(out);
    };
    let inverse = (1..prime)
        .find(|&z| {
            out.inverse_tests += 1;
            z as u32 * direction[2 * n + anchor] as u32 % prime as u32 == 1
        })
        .ok_or("missing nonzero field inverse")?;
    let point: Vec<_> = direction
        .iter()
        .map(|&x| (x as u32 * inverse as u32 % prime as u32) as u16)
        .collect();
    crate::checked_residuals(system, prime, &point)?;
    out.anchor_class = Some(anchor);
    out.inverse = Some(inverse);
    out.point = Some(point);
    out.reason = "normalized_direction".into();
    Ok(out)
}
pub(super) fn check(
    system: &System,
    n: usize,
    classes: usize,
    prime: u16,
    policy: finite_system::Policy,
    out: &Base,
) -> Result<()> {
    let problem = plan::field(system, prime, policy);
    if out.prime != prime || out.audit != finite_system::check(&problem, &out.field)? {
        return Err("categorical component field audit differs".into());
    }
    let Some(space) = &out.field.space else {
        if out.point.is_some()
            || out.anchor_class.is_some()
            || out.inverse.is_some()
            || out.inverse_tests != 0
            || out.reason != out.field.reason
        {
            return Err("incomplete field elimination supplied a code".into());
        }
        return Ok(());
    };
    if space.particular.iter().any(|&v| v != 0) {
        return Err("homogeneous canonical particular solution is nonzero".into());
    }
    if space.directions.is_empty() {
        if out.reason != "zero_kernel"
            || out.point.is_some()
            || out.anchor_class.is_some()
            || out.inverse.is_some()
            || out.inverse_tests != 0
        {
            return Err("zero kernel falsely supplied a nonconstant code".into());
        }
        return Ok(());
    }
    if space.directions.len() != 1 {
        return Err("unbounded component kernel was accepted".into());
    }
    let direction = &space.directions[0];
    let available: Vec<_> = (0..classes)
        .filter(|&i| direction[2 * n + i] != 0)
        .collect();
    if available.is_empty() {
        if out.reason != "class_constant_direction"
            || out.point.is_some()
            || out.anchor_class.is_some()
            || out.inverse.is_some()
            || out.inverse_tests != 0
        {
            return Err("input-only kernel direction became a class code".into());
        }
        return Ok(());
    }
    let anchor = out.anchor_class.ok_or("missing nonzero class anchor")?;
    let inverse = out.inverse.ok_or("missing class-anchor inverse")?;
    let point = out.point.as_ref().ok_or("missing normalized component")?;
    if anchor != available[0]
        || inverse == 0
        || inverse >= prime
        || out.inverse_tests != inverse as usize
        || out.reason != "normalized_direction"
        || point.len() != system.variables
        || (inverse as u64 * direction[2 * n + anchor] as u64) % prime as u64 != 1
    {
        return Err("invalid actual component unit normalization".into());
    }
    for (z, &x) in point.iter().zip(direction) {
        if *z as u64 != (inverse as u64 * x as u64) % prime as u64 {
            return Err("saved normalized direction differs".into());
        }
    }
    crate::checked_residuals(system, prime, point)?;
    if point[2 * n + anchor] != 1 {
        return Err("actual component anchor is not one".into());
    }
    Ok(())
}

use super::*;

pub(super) fn run(
    input: &Input,
    system: &System,
    factor: Factor,
    work: &mut Work,
    earlier_failed: bool,
) -> Result<Component> {
    let mut out = Component {
        factor,
        base: None,
        anchored_system: None,
        lifts: vec![],
        point: None,
        reason: if earlier_failed {
            "earlier_component_refused"
        } else {
            "stage_budget"
        }
        .into(),
    };
    if earlier_failed || work.finite_stages == input.policy.stages {
        return Ok(out);
    }
    let classes = system.variables - 2 * input.n;
    let base = base::run(
        system,
        input.n,
        classes,
        out.factor.prime,
        plan::stage_policy(&input.policy, work, 1),
    )?;
    plan::charge(work, &base.field);
    work.inverse_tests += base.inverse_tests;
    out.reason = base.reason.clone();
    let first = base.point.clone();
    let anchor = base.anchor_class;
    out.base = Some(base);
    let Some(mut x) = first else { return Ok(out) };
    work.normalization_products += system.variables;
    work.base_original_products += system.variables * system.rows.len();
    let anchored = plan::anchored(
        system,
        input.n,
        anchor.ok_or("normalized component lacks its anchor")?,
    );
    out.anchored_system = Some(anchored.clone());
    let mut modulus = out.factor.prime;
    for _ in 1..out.factor.exponent {
        if work.finite_stages == input.policy.stages {
            out.reason = "stage_budget".into();
            return Ok(out);
        }
        let step = lift::run(lift::Input {
            system: anchored.clone(),
            modulus,
            prime: out.factor.prime,
            base: x,
            policy: plan::stage_policy(&input.policy, work, 0),
        })?;
        plan::charge(work, &step.elimination);
        work.lift_original_products += step.work.base_products + step.work.next_products;
        let next = step.next.clone();
        out.reason = step.reason.clone();
        out.lifts.push(step);
        let Some(next) = next else { return Ok(out) };
        x = next;
        modulus *= out.factor.prime;
    }
    if modulus != out.factor.modulus {
        return Err("component exponent does not reach its stated modulus".into());
    }
    out.point = Some(x);
    out.reason = "complete".into();
    Ok(out)
}

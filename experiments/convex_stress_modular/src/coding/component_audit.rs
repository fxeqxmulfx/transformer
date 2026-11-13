use super::*;

fn empty(c: &Component, reason: &str) -> Result<()> {
    if c.base.is_some()
        || c.anchored_system.is_some()
        || !c.lifts.is_empty()
        || c.point.is_some()
        || c.reason != reason
    {
        return Err("skipped component contains work or an endpoint".into());
    }
    Ok(())
}
pub(super) fn check(
    input: &Input,
    system: &System,
    c: &Component,
    work: &mut Work,
    earlier_failed: bool,
) -> Result<()> {
    if earlier_failed {
        return empty(c, "earlier_component_refused");
    }
    if work.finite_stages == input.policy.stages {
        return empty(c, "stage_budget");
    }
    let base = c.base.as_ref().ok_or("missing attempted component base")?;
    base::check(
        system,
        input.n,
        system.variables - 2 * input.n,
        c.factor.prime,
        plan::stage_policy(&input.policy, work, 1),
        base,
    )?;
    plan::charge(work, &base.field);
    work.inverse_tests += base.inverse_tests;
    let Some(mut x) = base.point.clone() else {
        if c.reason != base.reason
            || c.anchored_system.is_some()
            || !c.lifts.is_empty()
            || c.point.is_some()
        {
            return Err("refused base became a component".into());
        }
        return Ok(());
    };
    work.normalization_products += system.variables;
    work.base_original_products += system.variables * system.rows.len();
    let anchored = plan::anchored(
        system,
        input.n,
        base.anchor_class
            .ok_or("missing actual normalized anchor")?,
    );
    if c.anchored_system.as_ref() != Some(&anchored) {
        return Err("component anchor changed the original system".into());
    }
    let mut modulus = c.factor.prime;
    for digit in 1..c.factor.exponent {
        if work.finite_stages == input.policy.stages {
            if c.lifts.len() != digit - 1 || c.point.is_some() || c.reason != "stage_budget" {
                return Err("stage-capped lift has extra state".into());
            }
            return Ok(());
        }
        let step = c
            .lifts
            .get(digit - 1)
            .ok_or("missing declared digit stage")?;
        let expected = lift::Input {
            system: anchored.clone(),
            modulus,
            prime: c.factor.prime,
            base: x,
            policy: plan::stage_policy(&input.policy, work, 0),
        };
        lift::check(&expected, step)?;
        plan::charge(work, &step.elimination);
        work.lift_original_products += step.work.base_products + step.work.next_products;
        let Some(next) = &step.next else {
            if c.lifts.len() != digit || c.point.is_some() || c.reason != step.reason {
                return Err("failed digit stage supplied further state".into());
            }
            return Ok(());
        };
        x = next.clone();
        modulus *= c.factor.prime;
    }
    if c.lifts.len() != c.factor.exponent - 1
        || c.point.as_ref() != Some(&x)
        || c.reason != "complete"
        || modulus != c.factor.modulus
    {
        return Err("component final lift chain, modulus or representative differs".into());
    }
    crate::checked_residuals(system, modulus, &x)?;
    Ok(())
}

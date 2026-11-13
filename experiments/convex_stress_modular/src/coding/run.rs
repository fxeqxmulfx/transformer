use super::*;

pub fn run(input: Input) -> Result<Report> {
    let buffer_bound_bytes = input.validate()?;
    let partition = plan::partition(&input);
    let q = partition.classes.len();
    let mut out = Report {
        input,
        buffer_bound_bytes,
        partition,
        system: None,
        factors: vec![],
        components: vec![],
        crt: vec![],
        crt_audits: vec![],
        point: None,
        collision: None,
        observation_conflict: None,
        model: None,
        audit: None,
        reason: "unsupported_latent_domain".into(),
        work: Work::default(),
    };
    if !(2..=257).contains(&q) {
        return Ok(out);
    }
    let system = plan::system(&out.input, &out.partition);
    system.validate()?;
    out.system = Some(system.clone());
    out.factors = plan::factors(q);
    let mut failed = false;
    for factor in &out.factors {
        let component = component::run(&out.input, &system, factor.clone(), &mut out.work, failed)?;
        failed |= component.point.is_none();
        out.components.push(component);
    }
    out.reason = "component_refused".into();
    if failed {
        return Ok(out);
    }
    let mut point = out.components[0]
        .point
        .clone()
        .ok_or("missing first component")?;
    let mut modulus = out.factors[0].modulus;
    for component in &out.components[1..] {
        let step = recombine::run(recombine::Input {
            system: system.clone(),
            left_modulus: modulus,
            right_modulus: component.factor.modulus,
            left: point,
            right: component
                .point
                .clone()
                .ok_or("missing complete component")?,
        })?;
        out.crt_audits.push(recombine::check(&step.input, &step)?);
        out.work.inverse_tests += step.inverse_tests;
        out.work.crt_original_products += step.original_products;
        point = step
            .point
            .clone()
            .ok_or("prime powers unexpectedly failed checked CRT")?;
        out.crt.push(step);
        modulus *= component.factor.modulus;
    }
    if modulus as usize != q {
        return Err("recombined modulus does not equal observed class count".into());
    }
    out.point = Some(point.clone());
    let codes = &point[2 * out.input.n..];
    let mut decoder = vec![None; q];
    for (class, &z) in codes.iter().enumerate() {
        if let Some(earlier) = decoder[z as usize] {
            out.collision = Some([earlier, class, z as usize]);
            out.reason = "class_code_collision".into();
            return Ok(out);
        }
        decoder[z as usize] = Some(class);
    }
    for (index, (&[cell, _], &class)) in out
        .input
        .observations
        .iter()
        .zip(&out.partition.class_indices)
        .enumerate()
    {
        out.work.observation_checks += 1;
        let z = (point[cell % out.input.n] as usize
            + point[out.input.n + cell / out.input.n] as usize)
            % q;
        if z != codes[class] as usize {
            out.observation_conflict = Some(index);
            out.reason = "original_observation_conflict".into();
            return Ok(out);
        }
    }
    let decoder = decoder
        .into_iter()
        .map(|c| {
            c.map(|i| out.partition.classes[i])
                .ok_or("uncovered class code")
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let n = out.input.n;
    let model = Model::construct(
        out.input.vocab,
        point[..n].iter().map(|&z| z as usize).collect(),
        point[n..2 * n].iter().map(|&z| z as usize).collect(),
        decoder,
    )?;
    let audit = model.audit(&out.input.samples())?;
    if audit["training_errors"] != 0 {
        return Err("categorical code admitted an observed error".into());
    }
    out.work.raw_frequencies = q;
    out.model = Some(model);
    out.audit = Some(audit);
    out.reason = "admitted".into();
    Ok(out)
}

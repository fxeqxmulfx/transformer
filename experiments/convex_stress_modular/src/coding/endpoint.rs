use super::*;

pub(super) fn check(input: &Input, out: &Report, point: &[u16], work: &mut Work) -> Result<()> {
    let q = out.partition.classes.len();
    let codes = &point[2 * input.n..];
    let mut collision = None;
    for later in 0..q {
        if let Some(earlier) = (0..later).find(|&i| codes[i] == codes[later]) {
            collision = Some([earlier, later, codes[later] as usize]);
            break;
        }
    }
    if out.collision != collision {
        return Err("actual class-code collision witness differs".into());
    }
    if collision.is_some() {
        if out.reason != "class_code_collision"
            || out.observation_conflict.is_some()
            || out.model.is_some()
            || out.audit.is_some()
        {
            return Err("colliding class codes supplied a decoder".into());
        }
        return Ok(());
    }
    let mut conflict = None;
    for (index, &[cell, label]) in input.observations.iter().enumerate() {
        work.observation_checks += 1;
        let class = out
            .partition
            .classes
            .iter()
            .position(|&c| c == label)
            .ok_or("original class absent")?;
        let sum = point[cell % input.n] as u64 + point[input.n + cell / input.n] as u64;
        if sum % q as u64 != codes[class] as u64 {
            conflict = Some(index);
            break;
        }
    }
    if out.observation_conflict != conflict {
        return Err("original observation-conflict witness differs".into());
    }
    if conflict.is_some() {
        if out.reason != "original_observation_conflict"
            || out.model.is_some()
            || out.audit.is_some()
        {
            return Err("endpoint violating original observations was admitted".into());
        }
        return Ok(());
    }
    let mut decoder = vec![];
    for z in 0..q {
        let class = codes
            .iter()
            .position(|&c| c as usize == z)
            .ok_or("missing decoder state")?;
        decoder.push(out.partition.classes[class]);
    }
    let n = input.n;
    let actual = Model::construct(
        input.vocab,
        point[..n].iter().map(|&z| z as usize).collect(),
        point[n..2 * n].iter().map(|&z| z as usize).collect(),
        decoder,
    )?;
    let audit = actual.audit(&input.samples())?;
    if out.model.as_ref() != Some(&actual)
        || out.audit.as_ref() != Some(&audit)
        || out.reason != "admitted"
        || audit["training_errors"] != 0
    {
        return Err(
            "actual categorical phase table or raw spectral/ordinary-CE audit differs".into(),
        );
    }
    work.raw_frequencies = q;
    Ok(())
}

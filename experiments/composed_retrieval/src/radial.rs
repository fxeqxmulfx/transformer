//! CM117. A bounded magnitude for each opaque input object in the product
//! channel. Destination probabilities and the two unrotated reads are shared
//! with the original complete forward. The classifier has no case argument.
use crate::{
    Result,
    model::{Model, Wave},
    operator::{Pass, Tables},
};

/// Evaluate the executed product from the original prefix and both actual
/// reads. Repeated objects contribute separately with their routing mass.
pub fn hidden(
    model: &Model,
    table: &Tables,
    pass: &Pass,
    prefix: &[usize],
    radii: &[f64],
) -> Result<Wave> {
    if radii.len() != model.vocabulary {
        return Err("radius table shape differs".into());
    }
    let Some((&query, memory)) = prefix.split_last() else {
        return Ok(pass.hidden);
    };
    if memory.is_empty() {
        return Ok(pass.hidden);
    }
    if memory.len() != pass.one.len() {
        return Err("inconsistent first-read path".into());
    }
    let radius = |i: usize| {
        radii
            .get(i)
            .copied()
            .filter(|r| (0.0..=1.0).contains(r))
            .ok_or("input object radius outside the unit interval")
    };
    let mut product_read = Wave::default();
    for (&id, &weight) in memory.iter().zip(&pass.one) {
        let r = radius(id)?;
        product_read.add(&table.values[id], weight * r);
    }
    let mut product = Wave::default();
    product.add(
        &product_read.times(&table.phases[query], false),
        radius(query)?,
    );
    let l = &table.layout;
    let base = Wave::mix(&pass.s1, &pass.s2, model.weights[l.depth]);
    Ok(Wave::mix(&base, &product, model.weights[l.product]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;

    #[test]
    fn unit_radii_preserve_all_paths_and_zero_blocks_either_operand() -> Result<()> {
        let mut m = Model::new(4, 7, &data::protocol()?, 2)?;
        let l = m.validate()?;
        m.weights[l.product] = 1.;
        let table = Tables::new(&m, &[1, 2, 3])?;
        let ones = [1.; 4];
        for prefix in [&[][..], &[0], &[2, 3], &[1, 0, 1], &[2, 3, 1, 1]] {
            let pass = table.pass(&m, prefix);
            let adjusted = hidden(&m, &table, &pass, prefix, &ones)?;
            assert_eq!(table.logits(&m, &adjusted), table.logits(&m, &pass.hidden));
        }
        let mut r = ones;
        r[2] = 0.;
        for pair in [[2, 1], [1, 2], [2, 2]] {
            let pass = table.pass(&m, &pair);
            assert_eq!(
                table.logits(&m, &hidden(&m, &table, &pass, &pair, &r)?),
                vec![0.; 4]
            );
        }
        r[2] = 0.5;
        let pass = table.pass(&m, &[2, 2]);
        let lower = table.logits(&m, &hidden(&m, &table, &pass, &[2, 2], &r)?);
        for (a, b) in lower.iter().zip(table.logits(&m, &pass.hidden)) {
            assert!((a - b / 4.).abs() < 1e-13);
        }
        Ok(())
    }
}

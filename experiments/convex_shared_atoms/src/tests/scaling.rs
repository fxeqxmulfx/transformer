//! Independent executed-tensor and finite-query checks for CM93's changed domain.
use super::*;

fn balanced(s: Shape) -> Result<Model> {
    let mut m = Model::balanced(s, 8.)?;
    m.mix(atom(s, 17)?, 0.3)?;
    m.mix(atom(s, 19)?, 0.2)?;
    Ok(m)
}

#[test]
fn every_scaled_complex_coordinate_matches_the_full_tensor() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let m = balanced(s)?;
    for length in 0..=3 {
        for mut word in 0..3usize.pow(length as u32) {
            let prefix: Vec<_> = (0..length)
                .map(|_| {
                    let t = (word % 3) as i64;
                    word /= 3;
                    t
                })
                .collect();
            let input = s.addresses(&prefix)?;
            for (c, actual) in m.complex(&input).iter().enumerate() {
                let mut expected = C::default();
                for &q in &input.queries {
                    for &(d, count) in &input.columns {
                        let entry: C = m
                            .atoms
                            .iter()
                            .zip(&m.weights)
                            .map(|(f, &w)| f.entry(c, q, d) * w)
                            .sum();
                        expected += C::new(1., 1.)
                            * entry
                            * count as f64
                            * m.query_scale[q]
                            * m.context_scale[d];
                    }
                }
                assert!((*actual - expected).norm() < 2e-14);
            }
            assert!(m.next_object(&prefix)? < s.objects);
        }
    }
    let mut invalid = m.clone();
    invalid.context_scale[1] = 0.;
    assert!(invalid.validate().is_err());
    invalid.context_scale[1] = f64::MIN_POSITIVE;
    invalid.query_scale[1] = f64::MIN_POSITIVE;
    assert!(invalid.validate().is_err());
    assert!(m.query_mass(8.).is_err());
    Ok(())
}

#[test]
fn scaled_queries_use_actual_reserved_mass_and_raw_ce_step() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let mut centre = balanced(s)?;
    let h = 1e-4;
    let mass = centre.query_mass(h)?;
    close(mass, 3. * h, 1e-18);
    let before = centre.logits(&[2, 1, 2])?;
    centre.shrink_for_queries(h)?;
    for (a, b) in centre.logits(&[2, 1, 2])?.iter().zip(before) {
        close(*a, (1. - mass / centre.budget) * b, 1e-14);
    }
    for prefix in [vec![], vec![1], vec![2, 1, 2]] {
        let base = centre.logits(&prefix)?;
        let measured = forward_scalars(&base, 2, h)?;
        for (c, &scalar) in measured.iter().enumerate() {
            let mut losses = vec![];
            for sign in [1., -1.] {
                let mut query = centre.clone();
                query.weights[0] -= mass;
                query.atoms.push(Atom::bias(s, c, sign)?);
                query.weights.push(mass);
                query.validate()?;
                let actual = query.logits(&prefix)?;
                for (j, &value) in actual.iter().enumerate() {
                    close(value, base[j] + if j == c { sign * h } else { 0. }, 2e-14);
                }
                losses.push(cross_entropy(&actual, 2)?);
            }
            close(scalar, (losses[0] - losses[1]) / (2. * h), 2e-11);
        }
    }
    Ok(())
}

#[test]
fn scaled_complete_gradient_keeps_complex_parts_duplicates_pairs_and_tail() -> Result<()> {
    let s = Shape::new(3, 4)?;
    let m = balanced(s)?;
    let records = [
        Record {
            prefix: &[0, 0, 0, 0, 1],
            target: 2,
        },
        Record {
            prefix: &[0, 0, 0, 0, 1],
            target: 2,
        },
        Record {
            prefix: &[2, 1],
            target: 0,
        },
    ];
    for batch in [&records[..], &records[2..]] {
        for mode in [Mode::Mini, Mode::ForwardMini] {
            let mut g = Gradient::zero(s)?;
            train::update_gradient(&mut g, &m, batch, config(mode))?;
            for d in 0..s.columns {
                for q in 0..s.queries {
                    for c in 0..s.objects {
                        for sign in [1., -1.] {
                            let h = 1e-5;
                            let mut difference = 0.;
                            for record in batch {
                                let input = s.addresses(record.prefix)?;
                                let mut logits = m.logits(record.prefix)?;
                                let count = input
                                    .columns
                                    .iter()
                                    .find(|&&(i, _)| i == d)
                                    .map_or(0, |&(_, n)| n);
                                let response = f64::from(input.queries.contains(&q))
                                    * count as f64
                                    * sign
                                    * m.query_scale[q]
                                    * m.context_scale[d];
                                logits[c] += h * response;
                                let plus = cross_entropy(&logits, record.target)?;
                                logits[c] -= 2. * h * response;
                                difference +=
                                    (plus - cross_entropy(&logits, record.target)?) / (2. * h);
                            }
                            close(
                                difference / batch.len() as f64,
                                sign * g.values[g.index(c, q, d)],
                                5e-9,
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn both_scaled_modes_preserve_capacity_and_checkpoint_geometry() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let records = [
        Record {
            prefix: &[0, 1, 2],
            target: 1,
        },
        Record {
            prefix: &[2, 2],
            target: 0,
        },
    ];
    let mut endpoints = vec![];
    for mode in [Mode::Mini, Mode::ForwardMini] {
        let mut m = balanced(s)?;
        let mut g = Gradient::zero(s)?;
        let mut rng = Random::new(52);
        for batch in [&records[..], &records[1..]] {
            let receipt = train::step(&mut m, &mut g, batch, config(mode), &mut rng)?;
            assert_eq!(receipt.records, batch.len());
            assert!(receipt.after_ce <= receipt.centre_ce);
        }
        let restored: Model = serde_json::from_slice(&serde_json::to_vec(&m)?)?;
        restored.validate()?;
        assert_eq!(restored.query_scale, m.query_scale);
        assert_eq!(restored.context_scale, m.context_scale);
        for record in &records {
            assert_eq!(
                m.logits(record.prefix)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>(),
                restored
                    .logits(record.prefix)?
                    .iter()
                    .map(|x| x.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        endpoints.push(m);
    }
    for record in &records {
        for (a, b) in endpoints[0]
            .logits(record.prefix)?
            .iter()
            .zip(endpoints[1].logits(record.prefix)?)
        {
            close(*a, b, 2e-6);
        }
    }
    Ok(())
}

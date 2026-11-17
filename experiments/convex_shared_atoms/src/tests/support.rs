use super::*;

#[test]
fn exact_support_preserves_dense_contractions_and_tiny_nonzero_rows() -> Result<()> {
    let s = Shape::new(3, 2)?;
    let mut g = Gradient::zero(s)?;
    let mut rng = Random::new(92);
    for (i, row) in g.values.chunks_exact_mut(s.objects).enumerate() {
        if i % 3 == 0 {
            for x in row {
                *x = rng.gauss(0., 1.);
            }
        }
    }
    g.values[s.objects] = 1e-270;
    let rows = g.active_rows();
    assert!(rows.contains(&1));
    assert!(!rows.contains(&2));
    let all: Vec<_> = (0..s.queries * s.columns).collect();
    for seed in [15, 72, 194] {
        let f = atom(s, seed)?;
        assert_eq!(
            g.value_rows(&f, &rows).to_bits(),
            g.value_rows(&f, &all).to_bits()
        );
        for axis in 0..3 {
            let compact = g.contract(&f, axis, &rows);
            let dense = g.contract(&f, axis, &all);
            for (a, b) in compact.iter().zip(dense) {
                assert_eq!(
                    (a.re.to_bits(), a.im.to_bits()),
                    (b.re.to_bits(), b.im.to_bits())
                );
            }
        }
    }
    // Refresh support after every update, retaining even tiny nonzero entries.
    g.values[2 * s.objects + 1] = -1e-260;
    g.values[s.objects] = 0.;
    let changed = g.active_rows();
    assert!(changed.contains(&2));
    assert!(!changed.contains(&1));
    let zero = Gradient::zero(s)?;
    assert!(zero.active_rows().is_empty());
    let (f, value) = zero.propose(&mut Random::new(7), 2, 4, None)?;
    assert_eq!(value, 0.);
    assert!(
        f.output
            .iter()
            .chain(&f.query)
            .chain(&f.context)
            .all(|&z| z == C::default())
    );
    Ok(())
}

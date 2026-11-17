use super::*;

#[test]
fn finite_ce_search_resolves_small_steps_under_large_candidate_scale() -> Result<()> {
    // Ten indexed records with both labels present. The optimal class-zero
    // logit is ln(7/3), independent of any optimizer implementation.
    let records: Vec<_> = (0..10)
        .map(|i| Record {
            prefix: &[],
            target: usize::from(i >= 7),
        })
        .collect();
    let base = vec![vec![0., 0.]; records.len()];
    let candidate = vec![vec![1e8, 0.]; records.len()];
    let (coarse_step, coarse_loss, _) = train::line_search(&base, &candidate, &records, 24)?;
    let (fine_step, fine_loss, _) = train::line_search(&base, &candidate, &records, 64)?;
    assert_eq!(coarse_step, 0.);
    close(coarse_loss, 2f64.ln(), 1e-14);
    assert!(fine_step > 0.);
    let optimum = -(0.7f64 * 0.7f64.ln() + 0.3f64 * 0.3f64.ln());
    close(fine_loss, optimum, 1e-10);
    close(fine_step * 1e8, (7f64 / 3.).ln(), 1e-5);
    println!(
        "coarse_step={coarse_step} coarse_ce={coarse_loss} fine_step={fine_step} fine_ce={fine_loss} analytic_optimum={optimum}"
    );
    Ok(())
}

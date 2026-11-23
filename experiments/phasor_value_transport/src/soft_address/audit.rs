//! Observe the real final-position queries while the frozen model generates.
use std::cell::RefCell;
#[derive(Default)]
struct Totals {
    rows: usize,
    near_certain: usize,
    clear: usize,
    min_mass: Option<f64>,
    gap: f64,
    mass: f64,
    rms: f64,
    entries: f64,
}
thread_local! {static TOTAL:RefCell<Option<Totals>>=const {RefCell::new(None)};}
pub fn start() {
    TOTAL.with(|t| *t.borrow_mut() = Some(Totals::default()));
}
pub(super) fn record(quality: Option<[f64; 4]>) {
    if let Some([entries, gap, mass, rms]) = quality {
        TOTAL.with(|cell| {
            if let Some(x) = cell.borrow_mut().as_mut() {
                x.rows += 1;
                x.entries += entries;
                x.mass += mass;
                x.gap += gap;
                x.rms += rms;
                x.near_certain += usize::from(mass >= 0.99);
                x.clear += usize::from(gap >= ((entries - 1.) * 100.).ln());
                x.min_mass = Some(x.min_mass.map_or(mass, |m| m.min(mass)));
            }
        });
    }
}
pub fn finish() -> serde_json::Value {
    let a = TOTAL
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_default();
    let mean = |v: f64| (a.rows > 0).then(|| v / a.rows as f64);
    serde_json::json!({"queries_with_two_writes":a.rows,"mean_entries":mean(a.entries),"mean_gap":mean(a.gap),
 "mean_top_mass":mean(a.mass),"minimum_top_mass":a.min_mass,"top_mass_at_least_99_percent":a.near_certain,
 "satisfy_99_percent_gap_bound":a.clear,"mean_soft_to_hard_payload_rms":mean(a.rms)})
}

//! Frozen hard reads and matched timings on learned final-prefix queries.
use std::cell::{Cell, RefCell};
thread_local! {
    static MODE:Cell<u8>=const {Cell::new(0)};
    static TOTALS:RefCell<Totals>=RefCell::new(Totals::default());
}
#[derive(Default)]
struct Totals {
    n: usize,
    disagreements: usize,
    clear: usize,
    clear_disagreements: usize,
    index_ns: u128,
    scan_ns: u128,
    soft_ns: u128,
    build_ns: u128,
    entries: usize,
}
pub fn enabled() -> bool {
    MODE.with(|m| m.get() != 0)
}
pub fn hard() -> bool {
    MODE.with(|m| m.get() == 2)
}
pub fn begin(hard: bool) {
    MODE.with(|m| m.set(if hard { 2 } else { 1 }));
    TOTALS.with(|v| *v.borrow_mut() = Totals::default());
}
pub(super) struct Timing {
    pub writes: usize,
    pub disagrees: bool,
    pub sharp: bool,
    pub index: u128,
    pub scan: u128,
    pub soft: u128,
    pub build: u128,
}
pub(super) fn observe(r: Timing) {
    TOTALS.with(|t| {
        let mut t = t.borrow_mut();
        t.n += 1;
        t.disagreements += usize::from(r.disagrees);
        t.clear += usize::from(r.sharp);
        t.clear_disagreements += usize::from(r.sharp && r.disagrees);
        t.index_ns += r.index;
        t.scan_ns += r.scan;
        t.soft_ns += r.soft;
        t.build_ns += r.build;
        t.entries += r.writes;
    });
}
pub fn finish() -> serde_json::Value {
    MODE.with(|m| m.set(0));
    TOTALS.with(|t|{
        let t=std::mem::take(&mut *t.borrow_mut());
        let mean=|x:u128|(t.n!=0).then(||x as f64/t.n as f64);
        serde_json::json!({"measured_queries":t.n,"index_vs_exhaustive_mismatches":t.disagreements,
            "clear_99_percent_gap":t.clear,"clear_gap_mismatches":t.clear_disagreements,
            "mean_writes":mean(t.entries as u128),"index_query_mean_ns":mean(t.index_ns),
            "exhaustive_hard_query_mean_ns":mean(t.scan_ns),"soft_payload_read_mean_ns":mean(t.soft_ns),
            "mean_total_index_build_ns_for_prefix":mean(t.build_ns)})
    })
}

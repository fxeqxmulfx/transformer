//! The hard-attention head that reads its keys as integers.
//!
//! `head.rs` answers a query by scoring the *stored* points, and that is where
//! `todo3.md` section 4's wall is: the compiler emits `ky = -k^2 + d`, the
//! ordinate is rounded on the way in, and past `k = ceil(sqrt(2^53))` -- about
//! 94.9 million -- two different keys arrive as one double.  No arbiter helps.
//! `exact::dot_cmp`, `long double` and `f128` all compare the points that were
//! stored, and none of them can recover a coordinate that was rounded before
//! it arrived.  `ALM.ScoreWall` is that sentence: the wall is the stored
//! coordinate, not the arithmetic.
//!
//! This head does not read the ordinate.  The abscissa `kx = 2k` is exact to
//! `k = 2^52`, and for a live key the ordinate is a *function* of it, so
//! `liftkey.rs` recovers `k` from `kx` and rebuilds the comparison in `i128`,
//! where `(k - q)^2` at the limit is `2^106` and an `i128` holds `2^127`.  The
//! wall moves from `2^26.5` to `2^52`, which is the whole point of this file.
//!
//! What licenses the container and the walk, theorem by theorem:
//!
//!   * The container is the sorted key set, one node per key.  `ALM.HullLift`
//!     says no erase rule fires on the paraboloid, `ALM.HullMark`'s
//!     `not_eraseStep_of_marked` that the recency term does not revive one,
//!     and `ALM.HullCover` that a hull holds one line per key -- so a plain
//!     ordered set of keys *is* the envelope, with no envelope code at all.
//!   * A query with `qy > 0` is concave in the key, so its maximum over the
//!     stored keys is at one of the two that straddle `qx / qy`, and a local
//!     maximum is the global one: `ALM.HullMark.not_eraseStep_of_marked`
//!     again, and `ALM.MarkedPosition.markKey_not_concurrent` that the offsets
//!     do not cost the strictness.
//!   * At most two keys can tie, and they are those two straddlers:
//!     `ALM.HullScan.argmaxSet_card_le_two`.
//!   * A query with `qy < 0` is convex in the key, so it is answered at one of
//!     the two ends: `ALM.HullLower`, that the lower half keeps two lines at
//!     any length.  Those are held in fields, as `head.rs` holds its own.
//!   * `qy == 0` reads one end and every key written there, offsets and all.
//!
//! The cleared entries are held *beside* the container rather than in it.
//! `clearkey.rs` says why they cannot go in it and `ALM.ClearKey` why they do
//! not have to: while the marker is wider than twice the largest base score,
//! a cleared entry loses to every live one at every query, so the answer is
//! the live container's answer and the marker is never scored at all.  When
//! that margin is not there -- `qy <= 0`, or a score past `BIG` itself, both
//! of which the new range reaches -- the cleared entries are asked in a hull
//! head of their own and the two winners are compared on their stored points,
//! which is neither better nor worse than what `head.rs` does today.
//!
//! And what is *not* covered leaves.  `lift.rs` names three families; only the
//! live one has an integer key with an offset under one, and only the cleared
//! one has the marker to hide behind.  The first key that is neither -- a flat
//! `ky = 1`, a non-integer `kx / 2`, a key past `2^52` -- retires the integer
//! path for good and hands the head to `HardAttentionHead`, rebuilt from the
//! journal.  On the released model that is the twenty flat heads immediately;
//! the remaining hundred and thirteen keep the integer path, and two of them
//! are heads `todo3.md` section 4b shows the old one answering wrongly.

use core::cell::Cell;
use core::cmp::Ordering;

use crate::breakpoint::Break;
use crate::clearkey::{ClearGuard, ClearKey};
use crate::exact;
use crate::grid::GridWitness;
use crate::head::HardAttentionHead;
use crate::liftkey::{LiftKey, UnitQuery};
use crate::meta::{HullMeta, TieBreak};
use crate::tree::{Line, Slope, Tree, NIL};

/// How much of a run the integer path carried.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct LiftCensus {
    /// Keys the integer path accepted.
    pub keys: usize,
    /// Keys carrying the clear marker, held beside the container rather than
    /// in it.
    pub cleared: usize,
    /// Queries answered by the `i128` comparison of `liftkey.rs`, which is
    /// exact to `2^52`.
    pub integer: usize,
    /// Of those, the ones whose abscissa the normalisation left off the
    /// integer, so the comparison carried the residual
    /// (`ALM.LiftResidual.upper_near_lt_iff`).  These are the queries that used
    /// to fall through to `stored`.
    pub near: usize,
    /// Queries answered by `exact::dot_cmp` on the reconstructed points: the
    /// keys are integers but the query is not on the unit grid, so the
    /// comparison is the stored one and the wall is back where it was.
    pub stored: usize,
    /// Queries answered along an axis, where the key's ordinate is not read.
    pub axis: usize,
    /// Queries handed to the hull head after the integer path retired.
    pub hull: usize,
    /// Queries a head holding cleared entries answered without looking at
    /// them, the marker being wider than twice any base score
    /// (`ALM.ClearKey.marked_sup'_eq_live`).
    pub dominated: usize,
    /// And queries where it was not, so the cleared entries had to be scored
    /// on their stored points beside the live ones.
    pub mixed: usize,
    /// Heads that retired, and the key that retired the first of them.
    pub retired: usize,
    pub retired_at: Option<[f64; 2]>,
}

impl LiftCensus {
    pub fn merge(&mut self, other: &LiftCensus) {
        self.keys += other.keys;
        self.cleared += other.cleared;
        self.integer += other.integer;
        self.near += other.near;
        self.stored += other.stored;
        self.axis += other.axis;
        self.hull += other.hull;
        self.dominated += other.dominated;
        self.mixed += other.mixed;
        self.retired += other.retired;
        self.retired_at = self.retired_at.or(other.retired_at);
    }
}

/// One extreme offset written at a key, and every entry carrying it.
///
/// Writing a key twice writes it with two offsets, and only one of them can
/// win: a query with `qy > 0` maximises `ky` and so takes the larger, one with
/// `qy < 0` minimises it and so takes the smaller, and they tie only when the
/// two offsets are the same double.  That is the rule `Envelope::add_line`
/// applies to two lines of equal slope, reached without building an envelope.
/// The tree node holds the larger, in place; `lower` keeps the smaller for the
/// two ends, which is all a `qy < 0` query ever reads.
type Peak = (f64, HullMeta);

fn lower(slot: &mut Peak, delta: f64, val: [f64; 2], seq: i32) {
    if delta < slot.0 {
        *slot = (delta, HullMeta::of(val, seq));
    } else if delta == slot.0 {
        slot.1.add(val, seq);
    }
}

/// The head: an ordered set of integer keys, and the hull head behind it.
pub struct LiftAttentionHead {
    /// One node per distinct key, ordered by it.  `m` is the key as a double,
    /// `b` the largest offset written at it, `meta` the entries carrying that
    /// offset.  The breakpoint field is never read: this tree is searched by
    /// key, and the query computes its own.
    live: Tree,
    /// Every key accepted so far, in arrival order, kept only while the
    /// integer path is live and dropped the moment it retires.
    journal: Vec<([f64; 2], [f64; 2], i32)>,
    /// Where the head goes when a key arrives that the argument above does not
    /// cover.
    other: Option<HardAttentionHead>,
    /// The cleared entries, in a head of their own, built only once one
    /// arrives.  `ALM.ClearKey.marked_sup'_eq_live` is what lets most queries
    /// skip it; `guard` is the hypothesis of that theorem, tested.
    cleared: Option<Box<HardAttentionHead>>,
    guard: ClearGuard,
    /// Every entry, for the one query that reads them all: `q == (0, 0)`.
    global: HullMeta,
    /// The extreme *live* key, which is where a `qy < 0` query is answered.
    min_v: i64,
    max_v: i64,
    /// The extreme abscissa over every entry, cleared ones included, and every
    /// entry written there.  That is what a query along the abscissa returns:
    /// the ordinate is multiplied by zero, so neither the offsets nor the
    /// marker separate anything and a cleared entry is an ordinary competitor.
    min_kx: f64,
    max_kx: f64,
    left_all: HullMeta,
    right_all: HullMeta,
    /// And the *smallest* offset at each end, which is what a query with
    /// `qy < 0` returns there: it maximises `-ky`.
    left_low: Peak,
    right_low: Peak,
    n: usize,
    /// Written from `query`, which takes `&self`: the census is an observation
    /// of the head, not part of its answer.
    census: Cell<LiftCensus>,
    /// And what it answered with no margin to spare, over every query whose
    /// answer came from a stored ordinate: the integer path is exact and has
    /// nothing to put here.
    grid: Cell<GridWitness>,
}

impl Default for LiftAttentionHead {
    fn default() -> Self {
        LiftAttentionHead {
            live: Tree::new(),
            journal: Vec::new(),
            other: None,
            cleared: None,
            guard: ClearGuard::default(),
            global: HullMeta::default(),
            min_v: i64::MAX,
            max_v: i64::MIN,
            min_kx: f64::INFINITY,
            max_kx: f64::NEG_INFINITY,
            left_all: HullMeta::default(),
            right_all: HullMeta::default(),
            left_low: (f64::INFINITY, HullMeta::default()),
            right_low: (f64::INFINITY, HullMeta::default()),
            n: 0,
            census: Cell::new(LiftCensus::default()),
            grid: Cell::new(GridWitness::default()),
        }
    }
}

impl LiftAttentionHead {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn clear(&mut self) {
        let (census, grid) = (self.census.get(), self.grid.get());
        *self = Self::default();
        self.census.set(census);
        self.grid.set(grid);
    }

    /// What this head carried, and what it handed away.
    pub fn census(&self) -> LiftCensus {
        self.census.get()
    }

    /// Whether the integer path is still the one answering.
    pub fn on_the_integers(&self) -> bool {
        self.other.is_none()
    }

    /// What this head has answered with no margin to spare.
    ///
    /// Three of the four paths a query can take report here, and the fourth is
    /// the one that has nothing to report: on the unit grid the comparison ran
    /// in `i128` with a whole unit of margin
    /// (`ALM.LiftCompare.upper_lt_iff`, `sq_dist_le`), so there is no rounding
    /// to measure and a silent witness is the claim rather than an omission.
    /// Off the grid the answer came from the stored points and the old wall is
    /// back (`ALM.ScoreWall`); along an axis the ordinate is multiplied by zero
    /// but the product `qx * kx` still rounds.  Both are measured, on the
    /// margin `head.rs` measures them on, so the two heads are comparable at
    /// every query neither of them answers as integers.
    ///
    /// The fourth contributor is the marker.  A query a cleared entry *wins*
    /// was decided in float64 at `10^30`, where `ulp` is `2^47` and the unit
    /// step is long gone (`ALM.ClearKey.the_marker_costs_the_grid`), so it is
    /// recorded -- and a query where the cleared entry merely took part and
    /// lost is not, because nothing about the answer rested on it.
    ///
    /// `LiftCensus::stored` counts the queries the second of these covers, and
    /// it is the number to read beside this one: a worst ratio of zero means
    /// something only once it is known how many queries were weighed for it.
    pub fn grid_witness(&self) -> GridWitness {
        let mut w = self.grid.get();
        if let Some(h) = self.other.as_ref() {
            w.merge(&h.grid_witness());
        }
        w
    }

    fn note(&self, f: impl FnOnce(&mut LiftCensus)) {
        let mut c = self.census.get();
        f(&mut c);
        self.census.set(c);
    }

    /// Weigh one answer that came from a stored ordinate against the margin it
    /// had to beat, exactly as `HardAttentionHead` weighs its own: for these
    /// keys the runner-up is an integer step away and a step in the score is
    /// `|qy|`, or `|qx|` where the query lies along the abscissa.
    fn note_grid(&self, score: f64, margin: f64, query: [f64; 2], key: Option<[f64; 2]>) {
        let mut w = self.grid.get();
        w.observe(score, margin, query, key);
        self.grid.set(w);
    }

    pub fn insert(&mut self, key: [f64; 2], val: [f64; 2], seq: i32) {
        self.n += 1;
        self.global.add(val, seq);
        if let Some(h) = self.other.as_mut() {
            h.insert(key, val, seq);
            return;
        }
        // The live family is taken out first, and `ClearKey::of` says why it
        // has to be: past `|k| = 7.1e14` a live `-k^2` is below the marker's
        // own threshold, and the new range reaches that.
        if let Some(lk) = LiftKey::of(key) {
            self.journal.push((key, val, seq));
            self.note(|c| c.keys += 1);
            self.guard.observe(key[0], key[1]);
            self.ends(key[0], val, seq);
            self.insert_live(lk, val, seq);
        } else if let Some(ck) = ClearKey::of(key) {
            self.journal.push((key, val, seq));
            self.note(|c| c.cleared += 1);
            self.guard.observe(key[0], ck.base_bound);
            self.ends(key[0], val, seq);
            self.cleared.get_or_insert_with(Default::default).insert(key, val, seq);
        } else {
            self.retire(key);
            let h = self.other.as_mut().expect("retire installs the hull head");
            h.insert(key, val, seq);
        }
    }

    /// Hold the extremes of the abscissa, over every entry the head takes.
    fn ends(&mut self, kx: f64, val: [f64; 2], seq: i32) {
        if kx < self.min_kx {
            self.min_kx = kx;
            self.left_all = HullMeta::default();
        }
        if kx == self.min_kx {
            self.left_all.add(val, seq);
        }
        if kx > self.max_kx {
            self.max_kx = kx;
            self.right_all = HullMeta::default();
        }
        if kx == self.max_kx {
            self.right_all.add(val, seq);
        }
    }

    /// Hand the head to the hull, which holds for any key whatsoever.
    ///
    /// The journal is replayed in arrival order, so the hull head sees exactly
    /// the sequence it would have seen had it been the head all along, and
    /// from here the two are the same object.
    fn retire(&mut self, at: [f64; 2]) {
        let mut h = HardAttentionHead::new();
        for (k, v, s) in self.journal.drain(..) {
            h.insert(k, v, s);
        }
        self.journal.shrink_to_fit();
        self.live.clear();
        self.cleared = None;
        self.other = Some(h);
        self.note(|c| {
            c.retired += 1;
            c.retired_at = c.retired_at.or(Some(at));
        });
    }

    fn insert_live(&mut self, lk: LiftKey, val: [f64; 2], seq: i32) {
        if lk.v < self.min_v {
            self.min_v = lk.v;
            self.left_low = (f64::INFINITY, HullMeta::default());
        }
        if lk.v == self.min_v {
            lower(&mut self.left_low, lk.delta, val, seq);
        }
        if lk.v > self.max_v {
            self.max_v = lk.v;
            self.right_low = (f64::INFINITY, HullMeta::default());
        }
        if lk.v == self.max_v {
            lower(&mut self.right_low, lk.delta, val, seq);
        }

        let m = Slope::new(lk.v as f64);
        let at = self.live.lower_bound_slope(m);
        if at != NIL && self.live.slope_of(at) == m {
            let stored = self.live.key(at).1;
            if lk.delta > stored {
                self.live.set_intercept(at, lk.delta);
                self.live.set_meta(at, HullMeta::of(val, seq));
            } else if lk.delta == stored {
                let mut merged = self.live.meta_of(at);
                merged.add(val, seq);
                self.live.set_meta(at, merged);
            }
        } else {
            let line = Line { m, b: lk.delta, p: Break::Unset, meta: HullMeta::of(val, seq) };
            self.live.insert_before(at, line);
        }
    }

    /// The key stored at a cursor, read back as an integer and an offset.
    fn key_at(&self, i: u32) -> LiftKey {
        let (m, b) = self.live.key(i);
        LiftKey { v: m.get() as i64, delta: b }
    }

    pub fn query(&self, q: [f64; 2], tb: TieBreak) -> Option<[f64; 2]> {
        if let Some(h) = self.other.as_ref() {
            self.note(|c| c.hull += 1);
            return h.query(q, tb);
        }
        if self.n == 0 {
            return None;
        }
        let (qx, qy) = (q[0], q[1]);
        if qy == 0.0 {
            self.note(|c| c.axis += 1);
            // The ordinate is multiplied by zero, so no offset and no marker
            // separates anything -- but `qx * kx` is still a product of two
            // doubles, and a step between two abscissae is `2 |qx|`.  `|qx|` is
            // the margin recorded, which is what `head.rs` records and is the
            // conservative half of the truth.  At `q == (0, 0)` every key
            // scores zero and the tie is exact, which `observe` reads off a
            // margin of zero and does not count.
            return Some(if qx > 0.0 {
                self.note_grid(qx * self.max_kx, qx, q, None);
                self.right_all.resolve(tb)
            } else if qx < 0.0 {
                self.note_grid(qx * self.min_kx, qx, q, None);
                self.left_all.resolve(tb)
            } else {
                self.global.resolve(tb)
            });
        }

        // On the unit grid the comparison is the integers'; off it the keys
        // are still integers but the query is not, and the honest comparison
        // is the one `head.rs` makes -- of the points as they were stored.
        // `LiftKey::point` rebuilds those bit for bit, by Sterbenz: `ky` is
        // within one of `-k^2`, so `ky + k * k` is exact.
        let unit = UnitQuery::of(q);
        self.note(|c| match unit {
            Some(u) => {
                c.integer += 1;
                if u.eps != 0.0 {
                    c.near += 1;
                }
            }
            None => c.stored += 1,
        });
        let cmp = |a: LiftKey, b: LiftKey| match unit {
            Some(u) => u.cmp(a, b),
            None => exact::dot_cmp(q, a.point(), b.point()),
        };

        let mut best: Option<(LiftKey, HullMeta)> = None;
        if qy < 0.0 {
            // Convex in the key, so the maximum is at an end: `ALM.HullLower`.
            let ends: [(i64, &Peak); 2] =
                [(self.min_v, &self.left_low), (self.max_v, &self.right_low)];
            for (v, slot) in ends {
                if slot.1.count == 0 {
                    continue;
                }
                fold(&mut best, LiftKey { v, delta: slot.0 }, slot.1, &cmp);
                if self.min_v == self.max_v {
                    break;
                }
            }
        } else {
            // Concave in the key, so the maximum is at a key straddling
            // `qx / qy`.  That ratio is one rounded division, and at the top
            // of the range an ulp of it is a whole unit, so the walk starts
            // two keys early and covers five: any off-by-one the division
            // could make is inside the window, and a local maximum of a
            // concave family is the global one.
            let t = qx / qy;
            let at = self.live.lower_bound_slope(Slope::new(t));
            let mut x = if at == NIL { self.live.last() } else { at };
            for _ in 0..2 {
                let p = self.live.prev(x);
                if p == NIL {
                    break;
                }
                x = p;
            }
            for _ in 0..5 {
                if x == NIL {
                    break;
                }
                fold(&mut best, self.key_at(x), self.live.meta_of(x), &cmp);
                x = self.live.next(x);
            }
        }
        // Off the unit grid the winner was chosen by comparing stored points,
        // so the wall that applies is the old one and the query has to be
        // weighed against it -- `LiftCensus::stored` counts these, and without
        // this the count stood beside a witness that had never looked at them.
        // On the grid there is nothing to weigh: `UnitQuery::cmp` ran in
        // `i128`.
        if unit.is_none() {
            if let Some((bk, _)) = best {
                let p = bk.point();
                self.note_grid(q[0] * p[0] + q[1] * p[1], qy, q, Some(p));
            }
        }
        self.with_cleared(q, tb, best)
    }

    /// Put the cleared entries back into the answer, or prove they are not in
    /// it.
    ///
    /// `ALM.ClearKey.marked_sup'_eq_live` is the fast path and needs both of
    /// its remaining hypotheses: a live entry, which is `best.is_some()`, and
    /// the margin `2M < qy * B`, which is `ClearGuard::dominated`.  With both
    /// the cleared head is not even searched -- the theorem says its maximum
    /// is below every live score, so it cannot be the answer and cannot tie
    /// one either, the domination in `cleared_lt_live` being strict.
    ///
    /// Without them the two winners are compared on their stored points.  That
    /// is the old arithmetic and the old wall, deliberately: a score carrying
    /// `BIG` has no unit grid left to be exact on
    /// (`ALM.ClearKey.the_marker_costs_the_grid`), so there is nothing better
    /// to do here than what `head.rs` already does.
    fn with_cleared(
        &self,
        q: [f64; 2],
        tb: TieBreak,
        best: Option<(LiftKey, HullMeta)>,
    ) -> Option<[f64; 2]> {
        let Some(h) = self.cleared.as_ref() else {
            return best.map(|(_, m)| m.resolve(tb));
        };
        if best.is_some() && self.guard.dominated(q) {
            self.note(|c| c.dominated += 1);
            return best.map(|(_, m)| m.resolve(tb));
        }
        self.note(|c| c.mixed += 1);
        let Some(hit) = h.hit(q, tb) else {
            return best.map(|(_, m)| m.resolve(tb));
        };
        let Some((bk, mut bm)) = best else {
            let mut w = self.grid.get();
            w.observe(hit.score, q[1], q, Some(hit.best_key));
            self.grid.set(w);
            return Some(hit.meta.resolve(tb));
        };
        let won = exact::dot_cmp(q, hit.best_key, bk.point());
        if won != Ordering::Less {
            let mut w = self.grid.get();
            w.observe(hit.score, q[1], q, Some(hit.best_key));
            self.grid.set(w);
        }
        match won {
            Ordering::Greater => Some(hit.meta.resolve(tb)),
            Ordering::Equal => {
                bm.merge(&hit.meta);
                Some(bm.resolve(tb))
            }
            Ordering::Less => Some(bm.resolve(tb)),
        }
    }
}

/// Fold one candidate into the running maximum, merging what ties with it.
fn fold(
    best: &mut Option<(LiftKey, HullMeta)>,
    k: LiftKey,
    m: HullMeta,
    cmp: &impl Fn(LiftKey, LiftKey) -> Ordering,
) {
    match best {
        None => *best = Some((k, m)),
        Some((bk, bm)) => match cmp(k, *bk) {
            Ordering::Greater => *best = Some((k, m)),
            Ordering::Equal => bm.merge(&m),
            Ordering::Less => {}
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lift::CLEAR_MARK;

    /// `embed_key`'s recency term, as `alm-compile` computes it.
    fn offset(pos: f64) -> f64 {
        0.3 * (1.0 / std::f64::consts::LN_2 - 1.0 / (pos + 2.0).ln())
    }

    /// A live key: `kx = 2k`, `ky = -k^2` and the offset of the position.
    fn key(k: f64, pos: f64) -> [f64; 2] {
        [2.0 * k, -(k * k) + offset(pos)]
    }

    /// Both heads, fed the same entries in the same order.
    fn pair(entries: &[(f64, f64, f64)]) -> (LiftAttentionHead, HardAttentionHead) {
        let (mut lift, mut hull) = (LiftAttentionHead::new(), HardAttentionHead::new());
        for (seq, &(k, pos, payload)) in entries.iter().enumerate() {
            let (kk, vv) = (key(k, pos), [payload, -payload]);
            lift.insert(kk, vv, seq as i32);
            hull.insert(kk, vv, seq as i32);
        }
        (lift, hull)
    }

    #[test]
    fn below_the_wall_it_answers_what_the_hull_head_answers() {
        // The keys are small enough that every score is an exact double, so
        // the two heads are comparing the same numbers and any disagreement
        // would be about which key the walk reached, not about arithmetic.
        let entries: Vec<(f64, f64, f64)> =
            (0..60).map(|i| ((i * 7 % 41) as f64 - 20.0, (3 * i + 1) as f64, i as f64)).collect();
        let (lift, hull) = pair(&entries);
        let mut asked = 0;
        for qi in -300..=300 {
            let qx = qi as f64 / 7.0;
            for qy in [1.0, -1.0, 3.0, -0.5, 0.0] {
                for tb in [TieBreak::Latest, TieBreak::Average] {
                    let q = [qx, qy];
                    assert_eq!(lift.query(q, tb), hull.query(q, tb), "at {q:?} {tb:?}");
                    asked += 1;
                }
            }
        }
        assert!(asked > 6_000, "only {asked} queries were compared");
        assert!(lift.on_the_integers(), "no key left the live family");
        assert_eq!(lift.census().keys, 60);
    }

    #[test]
    fn past_the_wall_it_separates_what_the_hull_head_cannot() {
        // `todo3.md` section 4b: `0x20202020` is one of the 32-bit values the
        // WASM heads are keyed on, so `k = 336860161` and `k^2 = 1.13e17`,
        // where a double is 16 apart.  The three scores here are `k^2`,
        // `k^2 - 1` and `k^2 - 1`, and the rounding of the stored ordinate is
        // worth eight of those units -- so the hull head does not merely tie
        // the winner with its neighbour, it ranks a neighbour above it, which
        // no tie-break recovers.
        let k = 336860161.0;
        let entries = [(k - 1.0, 5.0, 10.0), (k, 9.0, 20.0), (k + 1.0, 13.0, 30.0)];
        let (lift, hull) = pair(&entries);
        let q = [k, 1.0];
        assert_eq!(lift.query(q, TieBreak::Latest), Some([20.0, -20.0]), "the key queried");
        assert_ne!(hull.query(q, TieBreak::Latest), Some([20.0, -20.0]), "and what was shipped");
        assert_eq!(lift.census().integer, 1, "answered on the integers");
        assert_eq!(lift.census().stored, 0);
    }

    #[test]
    fn a_key_the_argument_does_not_cover_retires_the_integer_path() {
        // The flat key of `ALM.CumSum` and a key off the paraboloid are both
        // outside every family this head can speak for, and either one hands
        // it over for good.  What it answers afterwards is the hull head's
        // answer, entry for entry, including the entries that arrived before.
        for late in [[8.0, 1.0], [9.0, -20.25]] {
            let (mut lift, mut hull) = pair(&[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (5.0, 3.0, 5.0)]);
            assert!(lift.on_the_integers());
            lift.insert(late, [99.0, 0.0], 3);
            hull.insert(late, [99.0, 0.0], 3);
            assert!(!lift.on_the_integers(), "{late:?} retired it");
            assert_eq!(lift.census().retired_at, Some(late));
            for qi in -40..=40 {
                for qy in [1.0, -1.0, 0.0] {
                    let q = [qi as f64, qy];
                    assert_eq!(lift.query(q, TieBreak::Latest), hull.query(q, TieBreak::Latest));
                }
            }
            assert_eq!(lift.census().integer, 0, "and nothing was answered on the integers");
            assert!(lift.census().hull > 200);
        }
    }

    #[test]
    fn a_cleared_key_is_held_beside_the_container_and_does_not_retire_it() {
        // `ALM.ClearKey.marked_sup'_eq_live`: the marker is not a key the
        // integer path can hold, and it is not a key it has to hold either.
        // The head stays on the integers, the live answers are unchanged, and
        // the hull head -- which does score the marker -- agrees with all of
        // it, at `qy < 0` where the marker wins and at `qy == 0` where it ties.
        let (mut lift, mut hull) = pair(&[(1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (5.0, 3.0, 5.0)]);
        let marked = [8.0, -16.0 - CLEAR_MARK + offset(9.0)];
        lift.insert(marked, [99.0, 0.0], 3);
        hull.insert(marked, [99.0, 0.0], 3);
        assert!(lift.on_the_integers(), "the marker did not retire it");
        assert_eq!(lift.census().cleared, 1);
        assert_eq!(lift.census().retired, 0);
        for qi in -40..=40 {
            for qy in [1.0, -1.0, 0.0] {
                for tb in [TieBreak::Latest, TieBreak::Average] {
                    let q = [qi as f64, qy];
                    assert_eq!(lift.query(q, tb), hull.query(q, tb), "at {q:?} {tb:?}");
                }
            }
        }
        let c = lift.census();
        assert_eq!(c.integer + c.axis + c.stored, 486, "every query, and none on the hull");
        assert_eq!(c.dominated, 162, "the qy > 0 half never looked at the marker");
        assert_eq!(c.mixed, 162, "and the qy < 0 half had to");
    }

    #[test]
    fn a_cleared_key_that_outgrows_the_marker_is_scored_rather_than_assumed() {
        // The margin `2M < qy * BIG` is not free in the range `liftkey.rs`
        // opens: at `2^52` a live score is `2^106`, eighty times the marker.
        // So the head asks the cleared entries even at `qy > 0`, and the
        // answer is the hull head's -- the old wall, honestly reached.
        let k = 1e15;
        let (mut lift, mut hull) = pair(&[(k, 1.0, 7.0)]);
        let marked = [2.0, -1.0 - CLEAR_MARK];
        lift.insert(marked, [99.0, 0.0], 1);
        hull.insert(marked, [99.0, 0.0], 1);
        assert!(lift.on_the_integers());
        assert_eq!(lift.census().cleared, 1);
        let q = [k, 1.0];
        assert_eq!(lift.query(q, TieBreak::Latest), hull.query(q, TieBreak::Latest));
        assert_eq!(lift.census().dominated, 0, "there was no margin to spend");
        assert_eq!(lift.census().mixed, 1);
    }

    #[test]
    fn a_query_along_the_abscissa_reads_every_entry_at_the_end() {
        // With `qy == 0` the ordinate is multiplied away, so the offsets
        // separate nothing and the whole of the end ties -- both writes to
        // `k = 5`, and the average of their payloads.
        let (lift, hull) = pair(&[(5.0, 1.0, 4.0), (5.0, 2.0, 6.0), (2.0, 3.0, 1.0)]);
        let avg = TieBreak::Average;
        assert_eq!(lift.query([1.0, 0.0], avg), Some([5.0, -5.0]));
        assert_eq!(lift.query([1.0, 0.0], avg), hull.query([1.0, 0.0], avg));
        assert_eq!(lift.query([-1.0, 0.0], TieBreak::Latest), Some([1.0, -1.0]), "the other end");
        assert_eq!(lift.query([0.0, 0.0], avg), hull.query([0.0, 0.0], avg));
        assert_eq!(lift.census().axis, 4);
    }

    #[test]
    fn writing_a_key_twice_splits_it_between_the_two_halves() {
        // One key, two positions: the later write carries the larger offset.
        // A query with `qy > 0` maximises `ky` and gets it; one with `qy < 0`
        // minimises `ky` and gets the earlier write instead.  That is not a
        // tie-break -- under `Average` the two halves still answer with one
        // payload each, because the two writes are two distinct points.
        let (lift, hull) = pair(&[(3.0, 1.0, 7.0), (3.0, 40.0, 9.0)]);
        let avg = TieBreak::Average;
        assert_eq!(lift.query([6.0, 1.0], avg), Some([9.0, -9.0]), "the later write");
        assert_eq!(lift.query([6.0, -1.0], avg), Some([7.0, -7.0]), "the earlier one");
        for q in [[6.0, 1.0], [6.0, -1.0], [-2.0, 1.0], [11.0, -1.0]] {
            assert_eq!(lift.query(q, TieBreak::Average), hull.query(q, TieBreak::Average), "{q:?}");
        }
    }

    #[test]
    fn the_only_tie_a_live_head_can_have_is_the_pair_either_side_of_the_query() {
        // Two keys the same distance from the query and the same offset --
        // the same position, which the compiler only reaches by writing two
        // keys in one step.  `ALM.HullScan.argmaxSet_card_le_two` is that this
        // is the largest tie there is, and `Average` returns the mean.
        let (lift, hull) = pair(&[(4.0, 6.0, 2.0), (10.0, 6.0, 8.0), (1.0, 7.0, 100.0)]);
        let q = [7.0, 1.0];
        assert_eq!(lift.query(q, TieBreak::Average), Some([5.0, -5.0]), "the mean of 2 and 8");
        assert_eq!(lift.query(q, TieBreak::Latest), Some([8.0, -8.0]), "and the later of them");
        assert_eq!(lift.query(q, TieBreak::Average), hull.query(q, TieBreak::Average));
        assert_eq!(lift.query(q, TieBreak::Latest), hull.query(q, TieBreak::Latest));
    }

    #[test]
    fn an_empty_head_answers_nothing_and_a_cleared_one_goes_back_to_the_start() {
        let mut h = LiftAttentionHead::new();
        assert_eq!(h.query([1.0, 1.0], TieBreak::Latest), None);
        h.insert(key(3.0, 1.0), [5.0, 0.0], 0);
        assert_eq!(h.len(), 1);
        h.clear();
        assert!(h.is_empty());
        assert_eq!(h.query([1.0, 1.0], TieBreak::Latest), None);
        assert_eq!(h.census().keys, 1, "the census survives the clear");
    }

    #[test]
    fn a_query_off_the_unit_grid_is_answered_by_the_points_that_were_stored() {
        // The keys are integers and the query is not, so `UnitQuery::of`
        // refuses it and the comparison falls back to `exact::dot_cmp` on the
        // reconstructed points.  `LiftKey::point` rebuilds them bit for bit,
        // so that is the same comparison `head.rs` would have made.
        let (lift, hull) = pair(&[(0.0, 1.0, 0.0), (3.0, 2.0, 3.0), (7.0, 3.0, 7.0)]);
        let q = [4.5, 1.0];
        assert_eq!(lift.query(q, TieBreak::Latest), hull.query(q, TieBreak::Latest));
        assert_eq!(lift.census().stored, 1);
        assert_eq!(lift.census().integer, 0);
        assert_eq!(lift.query([9.0, 1.0], TieBreak::Latest), Some([7.0, -7.0]));
        assert_eq!(lift.census().integer, 1, "and an integer query is not");
    }

    #[test]
    fn the_census_of_two_heads_adds_up() {
        let mut a = LiftCensus { keys: 3, integer: 5, stored: 1, ..LiftCensus::default() };
        let b = LiftCensus {
            keys: 2,
            hull: 4,
            retired: 1,
            retired_at: Some([1.0, 2.0]),
            ..LiftCensus::default()
        };
        a.merge(&b);
        assert_eq!((a.keys, a.integer, a.stored, a.hull, a.retired), (5, 5, 1, 4, 1));
        assert_eq!(a.retired_at, Some([1.0, 2.0]));
    }
}

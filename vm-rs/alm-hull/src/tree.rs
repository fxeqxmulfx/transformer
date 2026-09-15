//! The envelope's container: a red-black tree in an arena, with cursors.
//!
//! The C++ keeps the envelope in one `std::multiset` and never moves an
//! element: `erase(it)` is amortized constant and hands back the successor,
//! `++it` and `--it` are pointer steps, so the erase loops in `add_line` cost
//! one descent for the whole call.  Stable Rust's `BTreeSet` gives none of
//! that -- `btree_cursors` is unstable (rust#107540) -- so a port written on
//! it re-descends for every neighbour and every removal, which measured 5.3x
//! slower than a plain vector on the order the traces actually have.
//!
//! This is that container, written out.  Nodes live in an arena addressed by
//! `u32`, so a link is four bytes rather than eight and the whole envelope is
//! three allocations that grow by doubling.  The arena is split by field, not
//! by node: a descent by slope touches `key` and `nav` and nothing else, 32
//! bytes against the 96 a node would be if it were one struct.  Searching
//! 1e6 keys laid out that way costs 220ns against 356ns for the same search
//! over whole lines.
//!
//! Where the split is drawn matters as much as that it is drawn, and the
//! sudoku trace is what draws it.  The walks read a line's `m` and `b`
//! together and never anything else; the aggregate behind the line is read
//! once per query and once per insertion onto an equal slope.  So `m` and `b`
//! share a sixteen-byte slot, which is one cache line for the pair, and the
//! forty bytes of aggregate they used to sit beside are their own array.
//! Splitting the aggregate off alone was measured first and gained nothing --
//! `m` and `b` were still two misses.  Both changes together took the hull
//! from 21.37s to 20.13s of a 42s run.
//!
//! `Transformer.ALM.TreeQuery` is the descent of `lower_bound_slope` in Lean,
//! with the balance factor taken from Batteries' red-black development.  What
//! is not there is the rebalancing: `fix_insert` and `fix_erase` below are
//! checked by `audit` in the tests and by nothing else, so the depth bound is
//! a theorem about trees that satisfy the invariant, not a proof that these
//! do.
//!
//! Index `0` is the sentinel.  It is its own black leaf, every empty child
//! points at it, and it exists so that the rebalancing cases can name the
//! parent of nothing without a special case.

use core::cmp::Ordering;

use crate::breakpoint::Break;
use crate::meta::HullMeta;

/// The empty child, and the cursor one past either end of the envelope.
pub const NIL: u32 = 0;

/// A slope, ordered totally so that it can key the tree.
///
/// Construction folds `-0.0` to `0.0`: the lower envelope stores negated keys,
/// so a zero slope arrives with either sign, and the two must name one line.
#[derive(Clone, Copy, Debug)]
pub struct Slope(f64);

impl Slope {
    pub fn new(m: f64) -> Slope {
        Slope(if m == 0.0 { 0.0 } else { m })
    }
    pub fn get(self) -> f64 {
        self.0
    }
}

impl PartialEq for Slope {
    fn eq(&self, o: &Self) -> bool {
        self.0.total_cmp(&o.0) == Ordering::Equal
    }
}
impl Eq for Slope {}
impl PartialOrd for Slope {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Slope {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// A node's place in the tree: three arena indices and a colour.
#[derive(Clone, Copy)]
struct Link {
    left: u32,
    right: u32,
    parent: u32,
    red: bool,
}

const BLACK_NIL: Link = Link { left: NIL, right: NIL, parent: NIL, red: false };

/// What a query descent reads at each node: the breakpoint it compares, and
/// the two children it chooses between.
///
/// They sit together because the descent needs both at every level and the
/// query descent is nearly the whole of the cost: the sudoku trace answers
/// 123.5 million of them against two million insertions.  Split across two
/// arrays each level costs two cache misses instead of one.
#[derive(Clone, Copy)]
struct Nav {
    p: Break,
    l: Link,
}

const NAV_NIL: Nav = Nav { p: Break::Unset, l: BLACK_NIL };

/// One line of the envelope: `y = m x + b`, optimal up to `p`.
#[derive(Clone, Copy, Debug)]
pub struct Line {
    pub m: Slope,
    pub b: f64,
    pub p: Break,
    pub meta: HullMeta,
}

/// The envelope, ordered by slope and searchable by breakpoint.
pub struct Tree {
    key: Vec<(Slope, f64)>,
    nav: Vec<Nav>,
    meta: Vec<HullMeta>,
    root: u32,
    /// The ends, held rather than walked to.
    ///
    /// Every arrival order the engine sees is near-sorted, so nearly every
    /// insertion lands at one end and nearly every erase leaves from one.
    /// With the ends in hand those calls answer without touching the tree at
    /// all, which is the whole of what a C++ hint insert buys.
    ends: (u32, u32),
    free: u32,
    len: usize,
}

impl Default for Tree {
    fn default() -> Tree {
        Tree::new()
    }
}

impl Tree {
    pub fn new() -> Tree {
        Tree {
            key: vec![(Slope(0.0), 0.0)],
            nav: vec![NAV_NIL],
            meta: vec![HullMeta::default()],
            root: NIL,
            ends: (NIL, NIL),
            free: NIL,
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn clear(&mut self) {
        self.key.truncate(1);
        self.nav.truncate(1);
        self.meta.truncate(1);
        self.nav[0] = NAV_NIL;
        self.root = NIL;
        self.ends = (NIL, NIL);
        self.free = NIL;
        self.len = 0;
    }

    /// The whole of the node at `i`: the line, its breakpoint, and the
    /// aggregate behind it.
    ///
    /// Three arrays, so three places; the walks that run in the hot path ask
    /// for `key` instead, which is the part of it they read.
    pub fn get(&self, i: u32) -> Line {
        let k = i as usize;
        let (m, b) = self.key[k];
        Line { m, b, p: self.nav[k].p, meta: self.meta[k] }
    }

    /// The line at `i` and nothing else: what both walks compare.
    ///
    /// One load, because `m` and `b` are adjacent.  `break_against` and
    /// `key_at` ask for exactly this pair and the envelope of the sudoku
    /// trace is a 1.6 GB arena in which a node visit is a cache miss, so
    /// whether the pair is one slot or two is the difference the walks feel.
    #[inline(always)]
    pub fn key(&self, i: u32) -> (Slope, f64) {
        self.key[i as usize]
    }

    /// The aggregate behind the line at `i`.
    #[inline(always)]
    pub fn meta_of(&self, i: u32) -> HullMeta {
        self.meta[i as usize]
    }

    pub fn slope_of(&self, i: u32) -> Slope {
        self.key[i as usize].0
    }

    pub fn break_of(&self, i: u32) -> Break {
        self.nav[i as usize].p
    }

    pub fn set_break(&mut self, i: u32, p: Break) {
        self.nav[i as usize].p = p;
    }

    pub fn set_meta(&mut self, i: u32, meta: HullMeta) {
        self.meta[i as usize] = meta;
    }

    /// The links of the node at `i`, named rather than indexed.
    ///
    /// The bounds check stays.  Dropping it with `get_unchecked` was written
    /// and measured: the insert path did not move at all, and the query
    /// descent went from 1.284s to 1.216s over the four reference programs --
    /// 5%, for a container whose every cursor would then be an unchecked
    /// index.  Not a trade worth making.
    #[inline(always)]
    fn lk(&self, i: u32) -> &Link {
        &self.nav[i as usize].l
    }

    #[inline(always)]
    fn lk_mut(&mut self, i: u32) -> &mut Link {
        &mut self.nav[i as usize].l
    }

    /// The slope of the node at `i`, which is what a descent by slope reads.
    #[inline(always)]
    fn sl(&self, i: u32) -> Slope {
        self.key[i as usize].0
    }

    /// The breakpoint of the node at `i`, which is what a query reads.
    #[inline(always)]
    fn bk(&self, i: u32) -> Break {
        self.nav[i as usize].p
    }

    /// The leftmost node, or `NIL` when the envelope is empty.
    pub fn first(&self) -> u32 {
        self.ends.0
    }

    /// The rightmost node, or `NIL` when the envelope is empty.
    pub fn last(&self) -> u32 {
        self.ends.1
    }

    /// The next line along the envelope, or `NIL` past the end.
    ///
    /// The right end is answered from `ends` rather than walked off.  Without
    /// that test the last node, which by construction has no right child,
    /// climbs its whole parent chain to discover that there is nothing above
    /// it — a walk of `log n` loads from an arena that does not fit in cache,
    /// and one that `add_line` takes on every append and `HullHalf::query` on
    /// every query that lands at the end.
    pub fn next(&self, mut x: u32) -> u32 {
        if x == NIL || x == self.ends.1 {
            return NIL;
        }
        if self.lk(x).right != NIL {
            x = self.lk(x).right;
            while self.lk(x).left != NIL {
                x = self.lk(x).left;
            }
            return x;
        }
        let mut p = self.lk(x).parent;
        while p != NIL && self.lk(p).right == x {
            x = p;
            p = self.lk(p).parent;
        }
        p
    }

    /// The previous line along the envelope, or `NIL` before the start.
    ///
    /// The left end is answered from `ends`, for the reason `next` gives, and
    /// it is the hotter of the two: over the `sudoku` trace 82.8 % of all
    /// insertions arrive below every slope present, and each of them asks this
    /// of the first node.
    pub fn prev(&self, mut x: u32) -> u32 {
        if x == NIL {
            return self.ends.1;
        }
        if x == self.ends.0 {
            return NIL;
        }
        if self.lk(x).left != NIL {
            x = self.lk(x).left;
            while self.lk(x).right != NIL {
                x = self.lk(x).right;
            }
            return x;
        }
        let mut p = self.lk(x).parent;
        while p != NIL && self.lk(p).left == x {
            x = p;
            p = self.lk(p).parent;
        }
        p
    }

    /// The first line whose slope is not below `m`, or `NIL` past the end.
    ///
    /// The two ends are answered from the cached cursors, without a descent:
    /// that is the case the traces are almost entirely made of.
    ///
    /// `ALM.TreeQuery.lowerBound` is the loop below, accumulator and all, and
    /// `lowerBound_eq_find` is that it returns the first line the test accepts
    /// when the test rises along the envelope.  It costs one comparison per
    /// node on the path, so at most the tree's depth
    /// (`ALM.TreeQuery.lbCount_le_depth`), which balance puts at
    /// `2 log2(n + 1)` (`lbCount_le_two_log`).  That is twice what
    /// `ALM.BinSearch.bcount_le_log` prices the C++ array search at, and
    /// `ALM.TreeQuery.log_succ_bound` is that the factor is two and no more:
    /// the price of keeping the cursor semantics, paid once per query.
    pub fn lower_bound_slope(&self, m: Slope) -> u32 {
        let (lo, hi) = self.ends;
        if hi == NIL || self.sl(hi) < m {
            return NIL;
        }
        if self.sl(lo) >= m {
            return lo;
        }
        let (mut x, mut at) = (self.root, NIL);
        while x != NIL {
            if self.sl(x) < m {
                x = self.lk(x).right;
            } else {
                at = x;
                x = self.lk(x).left;
            }
        }
        at
    }

    /// The first line whose breakpoint reaches `x`, or `NIL` past the end.
    ///
    /// This is the heterogeneous search: the tree is ordered by slope, and it
    /// answers by breakpoint because the two orders agree along the envelope.
    pub fn lower_bound_break(&self, p: Break) -> u32 {
        let (lo, hi) = self.ends;
        if hi == NIL || self.bk(hi) < p {
            return NIL;
        }
        if self.bk(lo) >= p {
            return lo;
        }
        let (mut x, mut at) = (self.root, NIL);
        while x != NIL {
            if self.bk(x) < p {
                x = self.lk(x).right;
            } else {
                at = x;
                x = self.lk(x).left;
            }
        }
        at
    }

    /// Take a node off the free list, or grow the arena by one.
    fn alloc(&mut self, line: Line) -> u32 {
        let i = if self.free != NIL {
            let i = self.free;
            self.free = self.lk(i).parent;
            self.key[i as usize] = (line.m, line.b);
            self.nav[i as usize].p = line.p;
            self.meta[i as usize] = line.meta;
            i
        } else {
            let i = self.nav.len() as u32;
            self.key.push((line.m, line.b));
            self.nav.push(Nav { p: line.p, l: BLACK_NIL });
            self.meta.push(line.meta);
            i
        };
        *self.lk_mut(i) = Link { left: NIL, right: NIL, parent: NIL, red: true };
        i
    }

    /// Insert `line` immediately before `at`, or at the end when `at` is `NIL`.
    ///
    /// The caller has already found the place, so this costs no descent: the
    /// new node hangs off a leaf that the cursor names, and only the recolour
    /// walks upward -- amortized constant, the same shape as the C++ hint
    /// insert, which is why `add_line` here descends once and not twice.
    pub fn insert_before(&mut self, at: u32, line: Line) -> u32 {
        let x = self.alloc(line);
        self.len += 1;
        if self.root == NIL {
            self.root = x;
            self.lk_mut(x).red = false;
            self.ends = (x, x);
            return x;
        }
        let (parent, left) = if at == NIL {
            (self.ends.1, false)
        } else if self.lk(at).left == NIL {
            (at, true)
        } else {
            (self.prev(at), false)
        };
        if at == self.ends.0 {
            self.ends.0 = x;
        }
        if at == NIL {
            self.ends.1 = x;
        }
        self.lk_mut(x).parent = parent;
        if left {
            self.lk_mut(parent).left = x;
        } else {
            self.lk_mut(parent).right = x;
        }
        self.fix_insert(x);
        x
    }

    /// Unlink the node at `x` and return it to the free list.
    ///
    /// The cost that matters: the caller holds its successor from before the
    /// call, so a run of erases walks the envelope once rather than searching
    /// again for each one.  This is what `Vec::remove` cannot do and what
    /// `BTreeSet` cannot express on stable Rust.
    ///
    /// Every other cursor stays valid.  A node with two children is not filled
    /// in from its successor -- that would move a line to another cell and
    /// dangle the caller's handle on it -- the successor is relinked into the
    /// erased node's place instead, so `x` and only `x` is released.
    pub fn erase(&mut self, x: u32) {
        debug_assert!(x != NIL);
        self.len -= 1;
        if x == self.ends.0 {
            self.ends.0 = self.next(x);
        }
        if x == self.ends.1 {
            self.ends.1 = self.prev(x);
        }
        let (mut y, child) = if self.lk(x).left == NIL {
            (x, self.lk(x).right)
        } else if self.lk(x).right == NIL {
            (x, self.lk(x).left)
        } else {
            let mut s = self.lk(x).right;
            while self.lk(s).left != NIL {
                s = self.lk(s).left;
            }
            (s, self.lk(s).right)
        };
        let parent;
        if y != x {
            // Put `y` where `x` stood, taking over both of its children.
            let xl = self.lk(x).left;
            self.lk_mut(xl).parent = y;
            self.lk_mut(y).left = xl;
            if y != self.lk(x).right {
                parent = self.lk(y).parent;
                if child != NIL {
                    self.lk_mut(child).parent = parent;
                }
                self.lk_mut(parent).left = child;
                let xr = self.lk(x).right;
                self.lk_mut(y).right = xr;
                self.lk_mut(xr).parent = y;
            } else {
                parent = y;
            }
            let xp = self.lk(x).parent;
            if self.root == x {
                self.root = y;
            } else if self.lk(xp).left == x {
                self.lk_mut(xp).left = y;
            } else {
                self.lk_mut(xp).right = y;
            }
            self.lk_mut(y).parent = xp;
            let red = self.lk(y).red;
            self.lk_mut(y).red = self.lk(x).red;
            self.lk_mut(x).red = red;
            // The colour that left the tree is the one `y` carried, and `x`
            // now holds it: from here `y` names that node.
            y = x;
        } else {
            parent = self.lk(y).parent;
            if child != NIL {
                self.lk_mut(child).parent = parent;
            }
            if self.root == x {
                self.root = child;
            } else if self.lk(parent).left == x {
                self.lk_mut(parent).left = child;
            } else {
                self.lk_mut(parent).right = child;
            }
        }
        if !self.lk(y).red {
            self.fix_erase(child, parent);
        }
        *self.lk_mut(x) = Link { left: NIL, right: NIL, parent: self.free, red: false };
        self.free = x;
    }

    fn rotate_left(&mut self, x: u32) {
        let y = self.lk(x).right;
        let b = self.lk(y).left;
        self.lk_mut(x).right = b;
        if b != NIL {
            self.lk_mut(b).parent = x;
        }
        let p = self.lk(x).parent;
        self.lk_mut(y).parent = p;
        if p == NIL {
            self.root = y;
        } else if self.lk(p).left == x {
            self.lk_mut(p).left = y;
        } else {
            self.lk_mut(p).right = y;
        }
        self.lk_mut(y).left = x;
        self.lk_mut(x).parent = y;
    }

    fn rotate_right(&mut self, x: u32) {
        let y = self.lk(x).left;
        let b = self.lk(y).right;
        self.lk_mut(x).left = b;
        if b != NIL {
            self.lk_mut(b).parent = x;
        }
        let p = self.lk(x).parent;
        self.lk_mut(y).parent = p;
        if p == NIL {
            self.root = y;
        } else if self.lk(p).left == x {
            self.lk_mut(p).left = y;
        } else {
            self.lk_mut(p).right = y;
        }
        self.lk_mut(y).right = x;
        self.lk_mut(x).parent = y;
    }

    fn fix_insert(&mut self, mut x: u32) {
        while x != self.root && self.lk(self.lk(x).parent).red {
            let p = self.lk(x).parent;
            let g = self.lk(p).parent;
            let left = self.lk(g).left == p;
            let uncle = if left { self.lk(g).right } else { self.lk(g).left };
            if uncle != NIL && self.lk(uncle).red {
                self.lk_mut(p).red = false;
                self.lk_mut(uncle).red = false;
                self.lk_mut(g).red = true;
                x = g;
                continue;
            }
            let p = if left && self.lk(p).right == x {
                self.rotate_left(p);
                x
            } else if !left && self.lk(p).left == x {
                self.rotate_right(p);
                x
            } else {
                p
            };
            self.lk_mut(p).red = false;
            self.lk_mut(g).red = true;
            if left {
                self.rotate_right(g);
            } else {
                self.rotate_left(g);
            }
            break;
        }
        self.lk_mut(self.root).red = false;
    }

    fn fix_erase(&mut self, mut x: u32, mut parent: u32) {
        while x != self.root && (x == NIL || !self.lk(x).red) {
            let left = self.lk(parent).left == x;
            let mut w = if left { self.lk(parent).right } else { self.lk(parent).left };
            if w != NIL && self.lk(w).red {
                self.lk_mut(w).red = false;
                self.lk_mut(parent).red = true;
                if left {
                    self.rotate_left(parent);
                    w = self.lk(parent).right;
                } else {
                    self.rotate_right(parent);
                    w = self.lk(parent).left;
                }
            }
            if w == NIL {
                x = parent;
                parent = self.lk(x).parent;
                continue;
            }
            let (wl, wr) = (self.lk(w).left, self.lk(w).right);
            let red = |t: &Self, n: u32| n != NIL && t.nav[n as usize].l.red;
            if !red(self, wl) && !red(self, wr) {
                self.lk_mut(w).red = true;
                x = parent;
                parent = self.lk(x).parent;
                continue;
            }
            let (near, far) = if left { (wl, wr) } else { (wr, wl) };
            if !red(self, far) {
                self.lk_mut(near).red = false;
                self.lk_mut(w).red = true;
                if left {
                    self.rotate_right(w);
                    w = self.lk(parent).right;
                } else {
                    self.rotate_left(w);
                    w = self.lk(parent).left;
                }
            }
            self.lk_mut(w).red = self.lk(parent).red;
            self.lk_mut(parent).red = false;
            let far = if left { self.lk(w).right } else { self.lk(w).left };
            self.lk_mut(far).red = false;
            if left {
                self.rotate_left(parent);
            } else {
                self.rotate_right(parent);
            }
            x = self.root;
            parent = NIL;
        }
        if x != NIL {
            self.lk_mut(x).red = false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every red-black invariant, plus that the arena agrees with the order.
    ///
    /// Returns the black height so the recursion can compare siblings.
    fn check(t: &Tree, x: u32) -> usize {
        if x == NIL {
            return 1;
        }
        let l = t.nav[x as usize].l.left;
        let r = t.nav[x as usize].l.right;
        if l != NIL {
            assert_eq!(t.nav[l as usize].l.parent, x, "left child disowns its parent");
            assert!(t.key[l as usize].0 < t.key[x as usize].0, "left child is not below");
        }
        if r != NIL {
            assert_eq!(t.nav[r as usize].l.parent, x, "right child disowns its parent");
            assert!(t.key[x as usize].0 < t.key[r as usize].0, "right child is not above");
        }
        if t.nav[x as usize].l.red {
            assert!(l == NIL || !t.nav[l as usize].l.red, "red node with a red left child");
            assert!(r == NIL || !t.nav[r as usize].l.red, "red node with a red right child");
        }
        let (bl, br) = (check(t, l), check(t, r));
        assert_eq!(bl, br, "black heights differ under one node");
        bl + usize::from(!t.nav[x as usize].l.red)
    }

    fn audit(t: &Tree) {
        assert!(t.root == NIL || !t.nav[t.root as usize].l.red, "the root is red");
        assert_eq!(t.nav[t.root as usize].l.parent, NIL, "the root has a parent");
        check(t, t.root);
        // The cached ends must be the ends, or every fast path lies.
        let (mut lo, mut hi) = (t.root, t.root);
        while lo != NIL && t.nav[lo as usize].l.left != NIL {
            lo = t.nav[lo as usize].l.left;
        }
        while hi != NIL && t.nav[hi as usize].l.right != NIL {
            hi = t.nav[hi as usize].l.right;
        }
        assert_eq!(t.ends, (lo, hi), "the cached ends are not the ends");
        let mut n = 0;
        let mut x = t.first();
        let mut last = None;
        while x != NIL {
            if let Some(p) = last {
                assert!(t.slope_of(p) < t.slope_of(x), "the walk is not increasing");
                assert_eq!(t.next(p), x, "next disagrees with the walk");
                assert_eq!(t.prev(x), p, "prev disagrees with next");
            }
            last = Some(x);
            n += 1;
            x = t.next(x);
        }
        assert_eq!(n, t.len(), "the walk and the count disagree");
        assert_eq!(t.prev(NIL), last.unwrap_or(NIL), "prev(NIL) is not the last node");
    }

    fn line(m: f64) -> Line {
        Line { m: Slope::new(m), b: 0.0, p: Break::Unset, meta: HullMeta::default() }
    }

    /// Insert at the found place, which is how `add_line` uses the tree.
    fn insert(t: &mut Tree, m: f64) -> u32 {
        let at = t.lower_bound_slope(Slope::new(m));
        t.insert_before(at, line(m))
    }

    #[test]
    fn ordered_under_every_arrival_order() {
        for &step in &[1i64, -1, 7, -7, 1009] {
            let mut t = Tree::new();
            let mut want: Vec<i64> = Vec::new();
            for i in 0..400i64 {
                let k = (i * step).rem_euclid(401);
                if want.contains(&k) {
                    continue;
                }
                insert(&mut t, k as f64);
                want.push(k);
                audit(&t);
            }
            want.sort_unstable();
            let got: Vec<i64> = {
                let (mut v, mut x) = (Vec::new(), t.first());
                while x != NIL {
                    v.push(t.slope_of(x).get() as i64);
                    x = t.next(x);
                }
                v
            };
            assert_eq!(got, want, "step {step}");
        }
    }

    /// The cursor on a successor must survive erasing its predecessor: this is
    /// the whole reason the container exists, and the reason `erase` relinks.
    #[test]
    fn erase_keeps_the_successor_cursor() {
        let mut t = Tree::new();
        for i in 0..200 {
            insert(&mut t, f64::from(i));
        }
        // Drop every line from 40 up, walking forward exactly once.
        let mut x = t.lower_bound_slope(Slope::new(40.0));
        let mut dropped = 0;
        while x != NIL {
            let next = t.next(x);
            t.erase(x);
            dropped += 1;
            x = next;
        }
        assert_eq!(dropped, 160);
        audit(&t);
        assert_eq!(t.len(), 40);
        assert_eq!(t.slope_of(t.last()).get(), 39.0);
    }

    #[test]
    fn erase_from_both_ends_and_the_middle() {
        let mut t = Tree::new();
        let mut ids: Vec<u32> = Vec::new();
        for i in 0..300 {
            ids.push(insert(&mut t, f64::from(i)));
        }
        for (i, &id) in ids.iter().enumerate() {
            if i % 3 == 0 {
                t.erase(id);
                audit(&t);
            }
        }
        assert_eq!(t.len(), 200);
        let mut x = t.first();
        let mut want = 1;
        while x != NIL {
            assert_eq!(t.slope_of(x).get() as i32, want);
            want += if want % 3 == 2 { 2 } else { 1 };
            x = t.next(x);
        }
    }

    /// Erase everything, reinsert, and the arena must reuse its cells.
    #[test]
    fn the_free_list_is_reused() {
        let mut t = Tree::new();
        for i in 0..500 {
            insert(&mut t, f64::from(i));
        }
        let cells = t.nav.len();
        while t.first() != NIL {
            t.erase(t.first());
        }
        assert!(t.is_empty());
        audit(&t);
        for i in 0..500 {
            insert(&mut t, f64::from(-i));
        }
        assert_eq!(t.nav.len(), cells, "the arena grew instead of reusing");
        audit(&t);
    }

    /// A long interleaving of both operations, against a sorted vector.
    #[test]
    fn agrees_with_a_sorted_vector() {
        let mut t = Tree::new();
        let mut want: Vec<i64> = Vec::new();
        let mut rng = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for _ in 0..4000 {
            let k = (next() % 1000) as i64;
            match want.binary_search(&k) {
                Ok(i) => {
                    if next() % 2 == 0 {
                        let at = t.lower_bound_slope(Slope::new(k as f64));
                        t.erase(at);
                        want.remove(i);
                    }
                }
                Err(i) => {
                    insert(&mut t, k as f64);
                    want.insert(i, k);
                }
            }
        }
        audit(&t);
        let (mut got, mut x) = (Vec::new(), t.first());
        while x != NIL {
            got.push(t.slope_of(x).get() as i64);
            x = t.next(x);
        }
        assert_eq!(got, want);
    }

    /// The heterogeneous search: ordered by slope, answered by breakpoint.
    #[test]
    fn lower_bound_break_finds_the_owning_line() {
        let mut t = Tree::new();
        for i in 0..64i64 {
            let id = insert(&mut t, f64::from(i as i32));
            t.set_break(id, Break::between(i as f64, 0.0, (i + 1) as f64, -(i as f64) - 1.0));
        }
        t.set_break(t.last(), Break::PosInf);
        for i in 0..64i64 {
            let at = t.lower_bound_break(Break::between(0.0, 0.0, 1.0, -(i as f64) - 0.5));
            assert_ne!(at, NIL);
            assert!(t.break_of(at) >= Break::between(0.0, 0.0, 1.0, -(i as f64) - 0.5));
            let p = t.prev(at);
            assert!(p == NIL || t.break_of(p) < Break::between(0.0, 0.0, 1.0, -(i as f64) - 0.5));
        }
    }
}

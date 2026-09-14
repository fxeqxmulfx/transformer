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
//! three allocations that grow by doubling.  The arena is split by field:
//! a descent by slope touches `slope` and `link` and nothing else, 24 bytes a
//! node against the 96 a node would be if it were one struct.  Searching
//! 1e6 keys laid out that way costs 220ns against 356ns for the same search
//! over whole lines.
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
    slope: Vec<Slope>,
    brk: Vec<Break>,
    rest: Vec<(f64, HullMeta)>,
    link: Vec<Link>,
    root: u32,
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
            slope: vec![Slope(0.0)],
            brk: vec![Break::Unset],
            rest: vec![(0.0, HullMeta::default())],
            link: vec![BLACK_NIL],
            root: NIL,
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
        self.slope.truncate(1);
        self.brk.truncate(1);
        self.rest.truncate(1);
        self.link.truncate(1);
        self.link[0] = BLACK_NIL;
        self.root = NIL;
        self.free = NIL;
        self.len = 0;
    }

    /// The line at `i`, which is what a query and its neighbour walk read.
    pub fn get(&self, i: u32) -> Line {
        let k = i as usize;
        let (b, meta) = self.rest[k];
        Line { m: self.slope[k], b, p: self.brk[k], meta }
    }

    pub fn slope_of(&self, i: u32) -> Slope {
        self.slope[i as usize]
    }

    pub fn break_of(&self, i: u32) -> Break {
        self.brk[i as usize]
    }

    pub fn set_break(&mut self, i: u32, p: Break) {
        self.brk[i as usize] = p;
    }

    pub fn set_meta(&mut self, i: u32, meta: HullMeta) {
        self.rest[i as usize].1 = meta;
    }

    /// The leftmost node, or `NIL` when the envelope is empty.
    pub fn first(&self) -> u32 {
        let mut x = self.root;
        while x != NIL && self.link[x as usize].left != NIL {
            x = self.link[x as usize].left;
        }
        x
    }

    /// The rightmost node, or `NIL` when the envelope is empty.
    pub fn last(&self) -> u32 {
        let mut x = self.root;
        while x != NIL && self.link[x as usize].right != NIL {
            x = self.link[x as usize].right;
        }
        x
    }

    /// The next line along the envelope, or `NIL` past the end.
    pub fn next(&self, mut x: u32) -> u32 {
        if x == NIL {
            return NIL;
        }
        if self.link[x as usize].right != NIL {
            x = self.link[x as usize].right;
            while self.link[x as usize].left != NIL {
                x = self.link[x as usize].left;
            }
            return x;
        }
        let mut p = self.link[x as usize].parent;
        while p != NIL && self.link[p as usize].right == x {
            x = p;
            p = self.link[p as usize].parent;
        }
        p
    }

    /// The previous line along the envelope, or `NIL` before the start.
    pub fn prev(&self, mut x: u32) -> u32 {
        if x == NIL {
            return self.last();
        }
        if self.link[x as usize].left != NIL {
            x = self.link[x as usize].left;
            while self.link[x as usize].right != NIL {
                x = self.link[x as usize].right;
            }
            return x;
        }
        let mut p = self.link[x as usize].parent;
        while p != NIL && self.link[p as usize].left == x {
            x = p;
            p = self.link[p as usize].parent;
        }
        p
    }

    /// The first line whose slope is not below `m`, or `NIL` past the end.
    pub fn lower_bound_slope(&self, m: Slope) -> u32 {
        let (mut x, mut at) = (self.root, NIL);
        while x != NIL {
            let k = x as usize;
            if self.slope[k] < m {
                x = self.link[k].right;
            } else {
                at = x;
                x = self.link[k].left;
            }
        }
        at
    }

    /// The first line whose breakpoint reaches `x`, or `NIL` past the end.
    ///
    /// This is the heterogeneous search: the tree is ordered by slope, and it
    /// answers by breakpoint because the two orders agree along the envelope.
    pub fn lower_bound_break(&self, p: Break) -> u32 {
        let (mut x, mut at) = (self.root, NIL);
        while x != NIL {
            let k = x as usize;
            if self.brk[k] < p {
                x = self.link[k].right;
            } else {
                at = x;
                x = self.link[k].left;
            }
        }
        at
    }

    /// Take a node off the free list, or grow the arena by one.
    fn alloc(&mut self, line: Line) -> u32 {
        let i = if self.free != NIL {
            let i = self.free;
            self.free = self.link[i as usize].parent;
            self.slope[i as usize] = line.m;
            self.brk[i as usize] = line.p;
            self.rest[i as usize] = (line.b, line.meta);
            i
        } else {
            let i = self.link.len() as u32;
            self.slope.push(line.m);
            self.brk.push(line.p);
            self.rest.push((line.b, line.meta));
            self.link.push(BLACK_NIL);
            i
        };
        self.link[i as usize] = Link { left: NIL, right: NIL, parent: NIL, red: true };
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
            self.link[x as usize].red = false;
            return x;
        }
        let (parent, left) = if at == NIL {
            (self.last(), false)
        } else if self.link[at as usize].left == NIL {
            (at, true)
        } else {
            (self.prev(at), false)
        };
        self.link[x as usize].parent = parent;
        if left {
            self.link[parent as usize].left = x;
        } else {
            self.link[parent as usize].right = x;
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
        let (mut y, child) = if self.link[x as usize].left == NIL {
            (x, self.link[x as usize].right)
        } else if self.link[x as usize].right == NIL {
            (x, self.link[x as usize].left)
        } else {
            let mut s = self.link[x as usize].right;
            while self.link[s as usize].left != NIL {
                s = self.link[s as usize].left;
            }
            (s, self.link[s as usize].right)
        };
        let parent;
        if y != x {
            // Put `y` where `x` stood, taking over both of its children.
            let xl = self.link[x as usize].left;
            self.link[xl as usize].parent = y;
            self.link[y as usize].left = xl;
            if y != self.link[x as usize].right {
                parent = self.link[y as usize].parent;
                if child != NIL {
                    self.link[child as usize].parent = parent;
                }
                self.link[parent as usize].left = child;
                let xr = self.link[x as usize].right;
                self.link[y as usize].right = xr;
                self.link[xr as usize].parent = y;
            } else {
                parent = y;
            }
            let xp = self.link[x as usize].parent;
            if self.root == x {
                self.root = y;
            } else if self.link[xp as usize].left == x {
                self.link[xp as usize].left = y;
            } else {
                self.link[xp as usize].right = y;
            }
            self.link[y as usize].parent = xp;
            let red = self.link[y as usize].red;
            self.link[y as usize].red = self.link[x as usize].red;
            self.link[x as usize].red = red;
            // The colour that left the tree is the one `y` carried, and `x`
            // now holds it: from here `y` names that node.
            y = x;
        } else {
            parent = self.link[y as usize].parent;
            if child != NIL {
                self.link[child as usize].parent = parent;
            }
            if self.root == x {
                self.root = child;
            } else if self.link[parent as usize].left == x {
                self.link[parent as usize].left = child;
            } else {
                self.link[parent as usize].right = child;
            }
        }
        if !self.link[y as usize].red {
            self.fix_erase(child, parent);
        }
        self.link[x as usize] = Link { left: NIL, right: NIL, parent: self.free, red: false };
        self.free = x;
    }

    fn rotate_left(&mut self, x: u32) {
        let y = self.link[x as usize].right;
        let b = self.link[y as usize].left;
        self.link[x as usize].right = b;
        if b != NIL {
            self.link[b as usize].parent = x;
        }
        let p = self.link[x as usize].parent;
        self.link[y as usize].parent = p;
        if p == NIL {
            self.root = y;
        } else if self.link[p as usize].left == x {
            self.link[p as usize].left = y;
        } else {
            self.link[p as usize].right = y;
        }
        self.link[y as usize].left = x;
        self.link[x as usize].parent = y;
    }

    fn rotate_right(&mut self, x: u32) {
        let y = self.link[x as usize].left;
        let b = self.link[y as usize].right;
        self.link[x as usize].left = b;
        if b != NIL {
            self.link[b as usize].parent = x;
        }
        let p = self.link[x as usize].parent;
        self.link[y as usize].parent = p;
        if p == NIL {
            self.root = y;
        } else if self.link[p as usize].left == x {
            self.link[p as usize].left = y;
        } else {
            self.link[p as usize].right = y;
        }
        self.link[y as usize].right = x;
        self.link[x as usize].parent = y;
    }

    fn fix_insert(&mut self, mut x: u32) {
        while x != self.root && self.link[self.link[x as usize].parent as usize].red {
            let p = self.link[x as usize].parent;
            let g = self.link[p as usize].parent;
            let left = self.link[g as usize].left == p;
            let uncle = if left { self.link[g as usize].right } else { self.link[g as usize].left };
            if uncle != NIL && self.link[uncle as usize].red {
                self.link[p as usize].red = false;
                self.link[uncle as usize].red = false;
                self.link[g as usize].red = true;
                x = g;
                continue;
            }
            let p = if left && self.link[p as usize].right == x {
                self.rotate_left(p);
                x
            } else if !left && self.link[p as usize].left == x {
                self.rotate_right(p);
                x
            } else {
                p
            };
            self.link[p as usize].red = false;
            self.link[g as usize].red = true;
            if left {
                self.rotate_right(g);
            } else {
                self.rotate_left(g);
            }
            break;
        }
        self.link[self.root as usize].red = false;
    }

    fn fix_erase(&mut self, mut x: u32, mut parent: u32) {
        while x != self.root && (x == NIL || !self.link[x as usize].red) {
            let left = self.link[parent as usize].left == x;
            let mut w = if left {
                self.link[parent as usize].right
            } else {
                self.link[parent as usize].left
            };
            if w != NIL && self.link[w as usize].red {
                self.link[w as usize].red = false;
                self.link[parent as usize].red = true;
                if left {
                    self.rotate_left(parent);
                    w = self.link[parent as usize].right;
                } else {
                    self.rotate_right(parent);
                    w = self.link[parent as usize].left;
                }
            }
            if w == NIL {
                x = parent;
                parent = self.link[x as usize].parent;
                continue;
            }
            let (wl, wr) = (self.link[w as usize].left, self.link[w as usize].right);
            let red = |t: &Self, n: u32| n != NIL && t.link[n as usize].red;
            if !red(self, wl) && !red(self, wr) {
                self.link[w as usize].red = true;
                x = parent;
                parent = self.link[x as usize].parent;
                continue;
            }
            let (near, far) = if left { (wl, wr) } else { (wr, wl) };
            if !red(self, far) {
                self.link[near as usize].red = false;
                self.link[w as usize].red = true;
                if left {
                    self.rotate_right(w);
                    w = self.link[parent as usize].right;
                } else {
                    self.rotate_left(w);
                    w = self.link[parent as usize].left;
                }
            }
            self.link[w as usize].red = self.link[parent as usize].red;
            self.link[parent as usize].red = false;
            let far = if left { self.link[w as usize].right } else { self.link[w as usize].left };
            self.link[far as usize].red = false;
            if left {
                self.rotate_left(parent);
            } else {
                self.rotate_right(parent);
            }
            x = self.root;
            parent = NIL;
        }
        if x != NIL {
            self.link[x as usize].red = false;
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
        let l = t.link[x as usize].left;
        let r = t.link[x as usize].right;
        if l != NIL {
            assert_eq!(t.link[l as usize].parent, x, "left child disowns its parent");
            assert!(t.slope[l as usize] < t.slope[x as usize], "left child is not below");
        }
        if r != NIL {
            assert_eq!(t.link[r as usize].parent, x, "right child disowns its parent");
            assert!(t.slope[x as usize] < t.slope[r as usize], "right child is not above");
        }
        if t.link[x as usize].red {
            assert!(l == NIL || !t.link[l as usize].red, "red node with a red left child");
            assert!(r == NIL || !t.link[r as usize].red, "red node with a red right child");
        }
        let (bl, br) = (check(t, l), check(t, r));
        assert_eq!(bl, br, "black heights differ under one node");
        bl + usize::from(!t.link[x as usize].red)
    }

    fn audit(t: &Tree) {
        assert!(t.root == NIL || !t.link[t.root as usize].red, "the root is red");
        assert_eq!(t.link[t.root as usize].parent, NIL, "the root has a parent");
        check(t, t.root);
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
        let cells = t.link.len();
        while t.first() != NIL {
            t.erase(t.first());
        }
        assert!(t.is_empty());
        audit(&t);
        for i in 0..500 {
            insert(&mut t, f64::from(-i));
        }
        assert_eq!(t.link.len(), cells, "the arena grew instead of reusing");
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

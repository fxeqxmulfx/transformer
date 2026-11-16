//! CM70 baseline plus every exceptional eight-word Fourier column. Logical
//! support is independent of equal values, including all unknown-row classes.
use super::*;
use std::mem::size_of;

pub(super) const ALL: usize = usize::MAX;
pub(super) type Column = [f64; 8];

#[derive(Debug)]
pub(super) struct Block {
    pub baseline: Column,
    pub exception_start: usize,
    pub exception_len: usize,
    pub active_start: usize,
    pub active_len: usize,
}
#[derive(Debug)]
pub(super) struct Exception {
    pub class: u16,
    pub value: Column,
}
pub(super) fn same(a: &Column, b: &Column) -> bool {
    a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

pub(super) struct Support {
    known: [usize; 4],
    count: usize,
    all: bool,
    classes: usize,
}
impl Support {
    pub fn new(domain: &Domain, row: usize) -> Self {
        let mut out = Self {
            known: [0; 4],
            count: 0,
            all: false,
            classes: domain.shape.classes,
        };
        for label in &domain.labels[row..row + 4] {
            if let Some(c) = label {
                if !out.known[..out.count].contains(c) {
                    out.known[out.count] = *c;
                    out.count += 1;
                }
            } else {
                out.all = true;
            }
        }
        out.known[..out.count].sort_unstable();
        out
    }
    pub fn len(&self) -> usize {
        if self.all { self.classes } else { self.count }
    }
    pub fn all_classes(&self) -> bool {
        self.all
    }
    pub fn known(&self) -> &[usize] {
        &self.known[..self.count]
    }
    pub fn iter(&self) -> impl ExactSizeIterator<Item = usize> + '_ {
        (0..self.len()).map(|i| if self.all { i } else { self.known[i] })
    }
    fn contains(&self, c: usize) -> bool {
        if self.all {
            c < self.classes
        } else {
            self.known[..self.count].binary_search(&c).is_ok()
        }
    }
    // A class outside the observed labels captures the common unknown-row
    // contribution when it exists. Otherwise an actual active class suffices.
    fn baseline_index(&self) -> usize {
        if self.all {
            (0..self.classes)
                .find(|c| !self.known[..self.count].contains(c))
                .unwrap_or(0)
        } else {
            0
        }
    }
}

pub(super) struct Builder {
    head: Head,
    limit: u64,
    pub scratch: Vec<Column>,
}

fn grow<T>(
    values: &mut Vec<T>,
    required: usize,
    maximum: usize,
    other: u128,
    limit: u64,
) -> Result<()> {
    let old = values.capacity();
    if required <= old {
        return Ok(());
    }
    let peak = |n: usize| other + (old as u128 + n as u128) * size_of::<T>() as u128;
    let mut wanted = required.max(old.saturating_mul(2)).min(maximum);
    if peak(wanted) > limit as u128 {
        wanted = required;
    }
    if required > maximum || peak(wanted) > limit as u128 {
        return Err("complete head fallback and moving growth exceed admitted storage".into());
    }
    values.try_reserve_exact(wanted - values.len())?;
    if peak(values.capacity()) > limit as u128 {
        return Err("actual head reallocation exceeds admitted storage".into());
    }
    Ok(())
}

impl Builder {
    fn fixed(blocks: usize, classes: usize) -> u128 {
        size_of::<Self>() as u128
            + blocks as u128 * size_of::<Block>() as u128
            + classes as u128 * size_of::<Column>() as u128
            + size_of::<Support>() as u128
            + size_of::<Option<Construction>>() as u128
            + 2 * size_of::<Column>() as u128
    }
    pub fn new(shape: Shape, limit: u64) -> Result<Self> {
        shape.validate()?;
        if limit == 0
            || limit > 96_000_000_000
            || Self::fixed(shape.rows() / 4, shape.classes) > limit as u128
        {
            return Err("complete head metadata or scratch exceeds admitted storage".into());
        }
        let mut head = Head {
            shape,
            blocks: Vec::new(),
            exceptions: Vec::new(),
            active: Vec::new(),
            sharing: None,
        };
        head.blocks.try_reserve_exact(shape.rows() / 4)?;
        let mut scratch = Vec::new();
        scratch.try_reserve_exact(shape.classes)?;
        if scratch.capacity() != shape.classes
            || Self::fixed(head.blocks.capacity(), scratch.capacity()) > limit as u128
        {
            return Err("actual head metadata or scratch exceeds admitted storage".into());
        }
        scratch.resize(shape.classes, [0.; 8]);
        Ok(Self {
            head,
            limit,
            scratch,
        })
    }
    pub fn push(&mut self, support: &Support) -> Result<()> {
        if self.head.blocks.len() >= self.head.shape.rows() / 4 {
            return Err("too many complete head blocks".into());
        }
        let baseline = self.scratch[support.baseline_index()];
        let count = self.scratch[..support.len()]
            .iter()
            .filter(|x| !same(x, &baseline))
            .count();
        let fixed = Self::fixed(self.head.blocks.capacity(), self.scratch.capacity());
        let required = self.head.exceptions.len() + count;
        grow(
            &mut self.head.exceptions,
            required,
            self.head.shape.rows() / 4 * (self.head.shape.classes - 1),
            fixed + self.head.active.capacity() as u128 * size_of::<u16>() as u128,
            self.limit,
        )?;
        if !support.all {
            let required = self.head.active.len() + support.len();
            grow(
                &mut self.head.active,
                required,
                self.head.shape.rows() / 4 * 4.min(self.head.shape.classes),
                fixed + self.head.exceptions.capacity() as u128 * size_of::<Exception>() as u128,
                self.limit,
            )?;
        }
        let block = Block {
            baseline,
            exception_start: self.head.exceptions.len(),
            exception_len: count,
            active_start: if support.all {
                ALL
            } else {
                self.head.active.len()
            },
            active_len: support.len(),
        };
        for (i, c) in support.iter().enumerate() {
            if !same(&self.scratch[i], &baseline) {
                self.head.exceptions.push(Exception {
                    class: c as u16,
                    value: self.scratch[i],
                });
            }
            if !support.all {
                self.head.active.push(c as u16);
            }
        }
        self.head.blocks.push(block);
        Ok(())
    }
    pub fn finish(mut self) -> Result<Head> {
        if self.head.blocks.len() != self.head.shape.rows() / 4 {
            return Err("missing complete head blocks".into());
        }
        // The Fourier construction row dies before any grouping scratch is
        // allocated. Charge the entire existing head and builder stack too.
        drop(self.scratch);
        let room = crate::memory::remaining(
            self.limit,
            self.head.allocation_bytes() as u128 + size_of::<Self>() as u128,
        )?;
        self.head.sharing = Sharing::build(&self.head, room)?;
        Ok(self.head)
    }
}

impl Head {
    pub fn allocation_bytes(&self) -> u64 {
        (size_of::<Self>()
            + self.blocks.capacity() * size_of::<Block>()
            + self.exceptions.capacity() * size_of::<Exception>()
            + self.active.capacity() * size_of::<u16>()
            + self.sharing.as_ref().map_or(0, Sharing::heap_bytes)) as u64
    }
    pub(super) fn columns<'a>(&'a self, b: &'a Block) -> Columns<'a> {
        Columns {
            head: self,
            block: b,
            next: 0,
            exception: b.exception_start,
            selected: None,
        }
    }
    pub(super) fn computation_columns<'a>(&'a self, b: &'a Block) -> Columns<'a> {
        let mut columns = self.columns(b);
        if b.active_start == ALL {
            columns.selected = self.sharing.as_ref().map(Sharing::representatives);
        }
        columns
    }
    pub(super) fn column<'a>(&'a self, b: &'a Block, c: usize) -> &'a Column {
        let exceptions = &self.exceptions[b.exception_start..b.exception_start + b.exception_len];
        match exceptions.binary_search_by_key(&(c as u16), |x| x.class) {
            Ok(i) => &exceptions[i].value,
            Err(_) => &b.baseline,
        }
    }
    pub(super) fn validate_support(&self, domain: &Domain) -> Result<()> {
        if self.shape != domain.shape || self.blocks.len() != self.shape.rows() / 4 {
            return Err("head dimensions differ from full logical tensor".into());
        }
        let (mut active, mut exceptions) = (0, 0);
        for (bi, b) in self.blocks.iter().enumerate() {
            let support = Support::new(domain, bi * 4);
            let end = b
                .exception_start
                .checked_add(b.exception_len)
                .ok_or("invalid head offsets")?;
            let stored = self
                .exceptions
                .get(b.exception_start..end)
                .ok_or("invalid head exceptions")?;
            if b.exception_start != exceptions
                || b.active_len != support.len()
                || (b.active_start == ALL) != support.all
                || b.exception_len >= b.active_len
            {
                return Err("noncanonical complete head support".into());
            }
            if !support.all {
                let end = b
                    .active_start
                    .checked_add(b.active_len)
                    .ok_or("invalid active offsets")?;
                let ids = self
                    .active
                    .get(b.active_start..end)
                    .ok_or("invalid active classes")?;
                if b.active_start != active || !ids.iter().map(|&c| c as usize).eq(support.iter()) {
                    return Err("head omitted or reordered a logical class".into());
                }
                active = end;
            }
            if stored.windows(2).any(|w| w[0].class >= w[1].class)
                || stored
                    .iter()
                    .any(|e| !support.contains(e.class as usize) || same(&e.value, &b.baseline))
            {
                return Err("noncanonical head exceptions".into());
            }
            exceptions = end;
        }
        if active != self.active.len() || exceptions != self.exceptions.len() {
            return Err("unreferenced complete head storage".into());
        }
        if let Some(sharing) = &self.sharing {
            sharing.validate(self)?;
        }
        Ok(())
    }
}

pub(super) struct Columns<'a> {
    head: &'a Head,
    block: &'a Block,
    next: usize,
    exception: usize,
    selected: Option<&'a [u16]>,
}
impl<'a> Iterator for Columns<'a> {
    type Item = (usize, &'a Column);
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.selected.map_or(self.block.active_len, <[u16]>::len) {
            return None;
        }
        let c = if let Some(selected) = self.selected {
            selected[self.next] as usize
        } else if self.block.active_start == ALL {
            self.next
        } else {
            self.head.active[self.block.active_start + self.next] as usize
        };
        self.next += 1;
        let value = if self.exception < self.block.exception_start + self.block.exception_len
            && self.head.exceptions[self.exception].class as usize == c
        {
            self.exception += 1;
            &self.head.exceptions[self.exception - 1].value
        } else {
            &self.block.baseline
        };
        Some((c, value))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.selected.map_or(self.block.active_len, <[u16]>::len) - self.next;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Columns<'_> {}

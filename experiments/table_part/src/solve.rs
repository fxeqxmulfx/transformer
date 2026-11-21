//! Exact completion inside the class `g(a) + h(b) + m a b` over `F_p`, from labeled cells.

/// What labeled cells say about the class.
pub enum Completion {
    /// No function of the class takes these labels.
    NoMember,
    /// More than one function of the class takes them.
    NotDetermined,
    /// One function does: `value[a] − value[p + b] + m a b`.
    Determined { m: u32, value: Vec<u32> },
}

impl Completion {
    /// The completed label of the cell `(a, b)`, if there is one.
    pub fn predict(&self, p: u32, a: u32, b: u32) -> Option<u32> {
        let Self::Determined { m, value } = self else {
            return None;
        };
        Some((value[a as usize] + p - value[(p + b) as usize] + m * (a * b % p)) % p)
    }
}

fn inverse(x: u32, p: u32) -> u32 {
    let (mut out, mut base, mut e) = (1, x % p, p - 2);
    while e > 0 {
        if e & 1 == 1 {
            out = out * base % p;
        }
        base = base * base % p;
        e >>= 1;
    }
    out
}

/// The root of `x` and `value(x) − value(root) = constant + slope · m`. O(log p).
fn locate(parent: &[u32], constant: &[u32], slope: &[u32], p: u32, mut x: usize) -> (usize, u32, u32) {
    let (mut c, mut s) = (0, 0);
    while parent[x] as usize != x {
        c = (c + constant[x]) % p;
        s = (s + slope[x]) % p;
        x = parent[x] as usize;
    }
    (x, c, s)
}

/// Solves `value(a) − value(p + b) + m a b = y` on `cells`, with `m` unknown, by a union-find
/// whose offsets are affine in `m`. `present` marks the rows and columns of the table's domain.
/// O(|cells| log p) time, O(p) memory.
pub fn complete(p: u32, cells: &[[u32; 3]], present: &[bool]) -> Completion {
    let size = 2 * p as usize;
    let mut parent: Vec<u32> = (0..size as u32).collect();
    let mut weight = vec![1u32; size];
    let mut constant = vec![0u32; size];
    let mut slope = vec![0u32; size];
    let mut m = None;
    let mut components = present.iter().filter(|&&x| x).count();
    for &[a, b, y] in cells {
        let (row, column) = (a as usize, (p + b) as usize);
        let (root_a, c_a, s_a) = locate(&parent, &constant, &slope, p, row);
        let (root_b, c_b, s_b) = locate(&parent, &constant, &slope, p, column);
        let product = a * b % p;
        if root_a == root_b {
            // (c_a − c_b − y) + (s_a − s_b + a b) m = 0.
            let free = (c_a + p - c_b + p - y) % p;
            let coefficient = (s_a + p - s_b + product) % p;
            if coefficient == 0 {
                if free != 0 {
                    return Completion::NoMember;
                }
            } else {
                let forced = (p - free) * inverse(coefficient, p) % p;
                if *m.get_or_insert(forced) != forced {
                    return Completion::NoMember;
                }
            }
            continue;
        }
        // value(root_a) − value(root_b) = (y − c_a + c_b) + (s_b − s_a − a b) m.
        let link_c = (y + p - c_a + c_b) % p;
        let link_s = (s_b + p - s_a + p - product) % p;
        let (child, root, c, s) = if weight[root_a] <= weight[root_b] {
            (root_a, root_b, link_c, link_s)
        } else {
            (root_b, root_a, (p - link_c) % p, (p - link_s) % p)
        };
        parent[child] = root as u32;
        weight[root] += weight[child];
        constant[child] = c;
        slope[child] = s;
        components -= 1;
    }
    let Some(m) = m.filter(|_| components == 1) else {
        return Completion::NotDetermined;
    };
    let value = (0..size)
        .map(|x| {
            let (_, c, s) = locate(&parent, &constant, &slope, p, x);
            (c + s * m) % p
        })
        .collect();
    Completion::Determined { m, value }
}

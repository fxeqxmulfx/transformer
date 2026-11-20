//! The far rotations of ReRoPE and Leaky ReRoPE (Su, kexue.fm/archives/9708,
//! 计算), with LeRoPE's learned frequencies. Beyond the window the query at `t`
//! is rotated by `w + λ(t - w)` and the key at `key` by `λ key`, with
//! `λ = 1/k` (Leaky ReRoPE) or `λ = 0` (ReRoPE), so the pair's relative angle
//! is `w + λ(t - key - w)`: `Transformer.RoPE.ReRoPE.leaky_phase_attend_eq_onlineMerge`
//! and `rerope_phase_attend_eq_onlineMerge`.
//!
//! * `rotate`: gpt 5a37c51's `attention::rotate` with one table per part, the
//!   query's and the key's; the value is not rotated and not written, since
//!   the far pairs read only the far query and key. The adjoint adds `Rᵀ dy`
//!   to the near adjoint already in `out`.
//! * `frequency_backward`: gpt 5a37c51's `rotary::frequency_backward` with
//!   the angle of the part's own table. It always accumulates: it runs after
//!   the near pass of the same layer.
//! * `far_tables`: the base angles and their cosines and sines, in the layout
//!   and the host FP32 arithmetic of gpt's `rotary::base_tables`. The control's
//!   `rotary::table` kernel turns them into the learned rotations.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.f32(v, [a]);
    }
    fn store(p: u64, i: u32, v: f32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.f32([a], v);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn rotate<const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        rope_query: *const f32,
        rope_key: *const f32,
        head_dim: u32,
        heads: u32,
        length: u32,
        count: u32,
    ) {
        let (i, b, t, n, row, pos, part, c, pair, at, other, half, width): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        mul.lo.u32(width, heads, head_dim);
        mul.lo.u32(n, width, 3);
        div.u32(row, i, n);
        rem.u32(pos, row, length);
        div.u32(part, i, width);
        rem.u32(part, part, 3);
        rem.u32(c, i, head_dim);
        shr.u32(half, head_dim, 1);
        let value: pred;
        setp.eq.u32(value, part, 2);
        if value {
            ret();
        }
        let query: pred;
        setp.eq.u32(query, part, 0);
        let rope: u64;
        selp.u64(rope, rope_query, rope_key, query);
        let (v, z, co, si): (f32, f32, f32, f32);
        load(v, x, i);
        rem.u32(pair, c, half);
        mad.lo.u32(at, pos, half, pair);
        load(co, rope, at);
        mul.lo.u32(n, length, half);
        add.u32(at, at, n);
        load(si, rope, at);
        let low: pred;
        setp.lt.u32(low, c, half);
        add.u32(other, i, half);
        if !low {
            sub.u32(other, i, half);
        }
        load(z, x, other);
        mul.rn.f32(v, v, co);
        mul.rn.f32(z, z, si);
        if BACK {
            if !low {
                neg.f32(z, z);
            }
        } else {
            if low {
                neg.f32(z, z);
            }
        }
        add.rn.f32(v, v, z);
        if BACK {
            let near: f32;
            load(near, out, i);
            add.rn.f32(v, near, v);
        }
        store(out, i, v);
    }
    // One warp per shared band, as the control's; every far Q/K head in a
    // fixed order, accumulated into the same alpha vector after the near pass.
    pub unsafe extern "ptx-kernel" fn frequency_backward(
        gradient: *mut f32,
        rotated: *const f32,
        dy: *const f32,
        theta_query: *const f32,
        theta_key: *const f32,
        bands: u32,
        heads: u32,
        length: u32,
        rows: u32,
    ) {
        let (band, lane, term, total, twice_heads, row, within, part, head): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        let (width, d, stride, at, imaginary, pos, ti): (u32, u32, u32, u32, u32, u32, u32);
        let theta: u64;
        let query: pred;
        mov.u32(band, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(term, lane);
        mul.lo.u32(d, bands, 2);
        mul.lo.u32(width, d, heads);
        mul.lo.u32(stride, width, 3);
        mul.lo.u32(twice_heads, heads, 2);
        mul.lo.u32(total, rows, twice_heads);
        let (sum, qr, qi, dr, di, angle, value, tmp): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(sum, 0.0);
        let done: pred;
        label(next);
        setp.ge.u32(done, term, total);
        if done {
            bra(finish);
        }
        div.u32(row, term, twice_heads);
        rem.u32(within, term, twice_heads);
        div.u32(part, within, heads);
        rem.u32(head, within, heads);
        mad.lo.u32(at, row, stride, band);
        mad.lo.u32(at, part, width, at);
        mad.lo.u32(at, head, d, at);
        add.u32(imaginary, at, bands);
        load(qr, rotated, at);
        load(qi, rotated, imaginary);
        load(dr, dy, at);
        load(di, dy, imaginary);
        rem.u32(pos, row, length);
        mad.lo.u32(ti, pos, bands, band);
        setp.eq.u32(query, part, 0);
        selp.u64(theta, theta_query, theta_key, query);
        load(angle, theta, ti);
        mul.rn.f32(value, qr, di);
        mul.rn.f32(tmp, qi, dr);
        sub.rn.f32(value, value, tmp);
        fma.rn.f32(sum, value, angle, sum);
        add.u32(term, term, 32);
        bra(next);
        label(finish);
        for j in 0..5 {
            shfl.sync.bfly.b32(value, sum, 1 << j, 31, 0xffffffff);
            add.rn.f32(sum, sum, value);
        }
        setp.eq.u32(done, lane, 0);
        if done {
            load(tmp, gradient, band);
            add.rn.f32(sum, tmp, sum);
            store(gradient, band, sum);
        }
    }
}

/// The far base angles of the query and of the key, each with its cosines
/// and sines: `angles[pos * bands + band]` and `rope` = cosines then sines,
/// as gpt's `rotary::base_tables`. The query's position is `w + (t - w)/k`
/// (ReRoPE, `slope = None`: `w`), the key's `key/k` (ReRoPE: `0`); a query
/// before the window has no far pair, and its row is never read.
pub fn far_tables(
    length: usize,
    head_dim: usize,
    window: usize,
    slope: Option<usize>,
) -> [(Vec<f32>, Vec<f32>); 2] {
    let bands = head_dim / 2;
    let table = |position: &dyn Fn(usize) -> f32| {
        let mut angles = vec![0.; length * bands];
        let mut rope = vec![0.; 2 * angles.len()];
        for pos in 0..length {
            for band in 0..bands {
                let angle = position(pos) / 10000f32.powf(2. * band as f32 / head_dim as f32);
                let i = pos * bands + band;
                angles[i] = angle;
                rope[i] = angle.cos();
                rope[angles.len() + i] = angle.sin();
            }
        }
        (angles, rope)
    };
    let w = window as f32;
    match slope {
        None => [table(&|_| w), table(&|_| 0.)],
        Some(k) => [
            table(&|t| (t as f32 - w) / k as f32 + w),
            table(&|key| key as f32 / k as f32),
        ],
    }
}

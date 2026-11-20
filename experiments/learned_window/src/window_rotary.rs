//! The far rotations of a window w and a slope s that each head of each layer
//! learns (`Position::Learned`), with LeRoPE's learned frequencies θ·e^α. A
//! pair r apart sits at min(r, w) + s·max(r − w, 0)
//! (`Transformer.RoPE.LeReRoPE.windowPosition`); one pass scores a key beyond
//! the window from the query rotated to w + s(t − w) and the key to s·j
//! (`route_eq_windowPosition`).
//!
//! * `state`: each head's w = window·e^ω and s = 1/(1 + (slope − 1)e^(−σ)),
//!   with 1 − s, and the near test's integer window ⌈w⌉: an integer distance
//!   is below w exactly when it is below ⌈w⌉.
//! * `table`: every layer's far rotations, per head: the query of row t at
//!   min(t, w) + s·max(t − w, 0), the key of row j at s·j. A row t < w has no
//!   far pair, so its far query is never read.
//! * `rotate`: `far_rotary::rotate` (gpt 5a37c51's `attention::rotate`) with
//!   a table per head.
//! * `torques`: per head and band, the far rotations' torques
//!   τ = Im(conj(x̃)·dx̃), summed over the batch: A = Σ_t τ(query t),
//!   E = Σ_t (t − w)·τ(query t) and C = Σ_j j·τ(key j). A query row t < w
//!   and a key row with no far query have no far adjoint, so τ = 0 there.
//! * `finish`: the gradients Lean states (`hasDerivAt_window_chunk_window`,
//!   `hasDerivAt_window_chunk_slope`), ∂L/∂w = (1 − s)Σ θ̂A and
//!   ∂L/∂s = Σ θ̂(E + C) over the head's bands, through ω and σ; and the far
//!   part of the frequencies' gradient, θ̂ Σ_heads (wA + s(E + C))
//!   (`hasDerivAt_rotate_adjoint` at the far positions).
//!
//! FP32 only, with no host readback or atomics, as gpt's `rotary`.
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
    fn store_window(p: u64, i: u32, v: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        st.global.u32([a], v);
    }
    // gpt 5a37c51's `rotary::exponential`: a range-reduced ninth-order Taylor
    // polynomial on [-ln(2)/2, ln(2)/2], with the rounding residual `elo`.
    fn exponential(e: f32, elo: f32, alpha: f32) {
        let (n, r, p, scale, minus_n, tail): (f32, f32, f32, f32, f32, f32);
        mul.rn.f32(n, alpha, 1.4426950408889634);
        cvt.rni.f32.f32(n, n);
        neg.f32(minus_n, n);
        fma.rn.f32(r, minus_n, 0.693145751953125, alpha);
        fma.rn.f32(r, minus_n, 0.0000014286068203094172, r);
        mov.f32(p, 1.0 / 362880.0);
        fma.rn.f32(p, p, r, 1.0 / 40320.0);
        fma.rn.f32(p, p, r, 1.0 / 5040.0);
        fma.rn.f32(p, p, r, 1.0 / 720.0);
        fma.rn.f32(p, p, r, 1.0 / 120.0);
        fma.rn.f32(p, p, r, 1.0 / 24.0);
        fma.rn.f32(p, p, r, 1.0 / 6.0);
        fma.rn.f32(p, p, r, 0.5);
        mul.rn.f32(p, p, r);
        fma.rn.f32(p, p, r, r);
        ex2.approx.f32(scale, n);
        fma.rn.f32(e, scale, p, scale);
        sub.rn.f32(tail, scale, e);
        fma.rn.f32(elo, scale, p, tail);
    }
    // The learned frequency θ·e^α of a band, as `rotary::table` forms it.
    fn learned(v: f32, frequencies: u64, alpha: u64, band: u32) {
        let (frequency, a, e, elo): (f32, f32, f32, f32);
        load(frequency, frequencies, band);
        load(a, alpha, band);
        exponential(e, elo, a);
        mul.rn.f32(v, frequency, e);
        fma.rn.f32(v, frequency, elo, v);
    }
    // Im(conj(x)·dx) of the chunk whose real part is at `at`, as the
    // control's `rotary::frequency_backward` forms it.
    fn torque(v: f32, x: u64, dx: u64, at: u32, bands: u32) {
        let (xr, xi, dr, di, tmp): (f32, f32, f32, f32, f32);
        let imaginary: u32;
        add.u32(imaginary, at, bands);
        load(xr, x, at);
        load(xi, x, imaginary);
        load(dr, dx, at);
        load(di, dx, imaginary);
        mul.rn.f32(v, xr, di);
        mul.rn.f32(tmp, xi, dr);
        sub.rn.f32(v, v, tmp);
    }

    /// Head `i` (layer-major): w at `shape[i]`, s at `shape[count + i]`,
    /// 1 − s at `shape[2·count + i]`, ⌈w⌉ at `windows[i]`, from ω at
    /// `parameters[i]` and σ at `parameters[count + i]`. With
    /// u = odds·e^(−σ), s = 1/(1 + u) and 1 − s = u/(1 + u); `odds` is
    /// slope − 1, so σ = 0 gives s = 1/slope.
    pub unsafe extern "ptx-kernel" fn state(
        windows: *mut u32,
        shape: *mut f32,
        parameters: *const f32,
        window: f32,
        odds: f32,
        count: u32,
    ) {
        let (i, b, t, n, at, ceiling): (u32, u32, u32, u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (omega, sigma, e, elo, w, u, denominator, s, rest): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        load(omega, parameters, i);
        add.u32(at, i, count);
        load(sigma, parameters, at);
        exponential(e, elo, omega);
        mul.rn.f32(w, e, window);
        neg.f32(sigma, sigma);
        exponential(e, elo, sigma);
        mul.rn.f32(u, e, odds);
        add.rn.f32(denominator, u, 1.0);
        div.rn.f32(s, 1.0, denominator);
        div.rn.f32(rest, u, denominator);
        // Rounds up and saturates: a NaN window is 0, every pair far.
        cvt.rpi.u32.f32(ceiling, w);
        store_window(windows, i, ceiling);
        store(shape, i, w);
        store(shape, at, s);
        add.u32(at, at, count);
        store(shape, at, rest);
    }

    /// Every layer's far rotations. A layer holds `block = 2·heads·length·bands`
    /// cosines, at `((part·heads + head)·length + pos)·bands + band` with the
    /// query's part 0 and the key's part 1, then as many sines. The angle of
    /// the position p is p·θ·e^α, formed as `rotary::table` forms its angles,
    /// reduced by the control's two-term 2π and turned by `sin.approx` and
    /// `cos.approx`. The control turns a host-precise base rotation by the
    /// small angle that α adds; p is learned here, so there is no host base.
    /// `stride` is the number of heads of all layers: the slope of head `i` is
    /// `shape[stride + i]`.
    pub unsafe extern "ptx-kernel" fn table(
        out: *mut f32,
        shape: *const f32,
        frequencies: *const f32,
        alpha: *const f32,
        heads: u32,
        length: u32,
        bands: u32,
        stride: u32,
        count: u32,
    ) {
        let (i, b, t, n, band, pos, head, part, layer, rest, block, at): (
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
        rem.u32(band, i, bands);
        div.u32(rest, i, bands);
        rem.u32(pos, rest, length);
        div.u32(rest, rest, length);
        rem.u32(head, rest, heads);
        div.u32(rest, rest, heads);
        and.b32(part, rest, 1);
        shr.u32(layer, rest, 1);
        let (w, s, position, excess, p, frequency, a, e, elo, base, angle): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (turns, negative_turns, si, co): (f32, f32, f32, f32);
        mad.lo.u32(at, layer, heads, head);
        load(w, shape, at);
        add.u32(at, at, stride);
        load(s, shape, at);
        cvt.rn.f32.u32(position, pos);
        // The key at s·j; the query at w + s(t − w) beyond the window and at
        // t before it.
        mul.rn.f32(p, s, position);
        let (query, inside): (pred, pred);
        setp.eq.u32(query, part, 0);
        if query {
            sub.rn.f32(excess, position, w);
            fma.rn.f32(p, s, excess, w);
            setp.lt.f32(inside, position, w);
            selp.f32(p, position, p, inside);
        }
        // The base angle p·θ, then `rotary::table`'s θ·e^α: angle·e + angle·elo.
        load(frequency, frequencies, band);
        load(a, alpha, band);
        exponential(e, elo, a);
        mul.rn.f32(base, p, frequency);
        mul.rn.f32(angle, base, e);
        fma.rn.f32(angle, base, elo, angle);
        mul.rn.f32(turns, angle, 0.15915494309189535);
        cvt.rni.f32.f32(turns, turns);
        neg.f32(negative_turns, turns);
        fma.rn.f32(angle, negative_turns, 6.28125, angle);
        fma.rn
            .f32(angle, negative_turns, 0.0019353071795864769, angle);
        sin.approx.f32(si, angle);
        cos.approx.f32(co, angle);
        mul.lo.u32(block, heads, length);
        mul.lo.u32(block, block, bands);
        shl.b32(block, block, 1);
        rem.u32(at, i, block);
        mul.lo.u32(rest, layer, block);
        mad.lo.u32(at, rest, 2, at);
        store(out, at, co);
        add.u32(at, at, block);
        store(out, at, si);
    }

    /// `far_rotary::rotate` with a table per head: the query and key parts of
    /// head `h` turn by the cosines and sines of `table`'s parts 0 and 1. The
    /// value is not rotated and not written. The adjoint adds `Rᵀ dy` to the
    /// near adjoint already in `out`.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn rotate<const BACK: bool>(
        out: *mut f32,
        x: *const f32,
        rope: *const f32,
        head_dim: u32,
        heads: u32,
        length: u32,
        count: u32,
    ) {
        let (i, b, t, n, row, pos, part, c, pair, at, other, half, width, head): (
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
        rem.u32(head, i, width);
        div.u32(head, head, head_dim);
        let (v, z, co, si): (f32, f32, f32, f32);
        load(v, x, i);
        rem.u32(pair, c, half);
        mad.lo.u32(at, part, heads, head);
        mad.lo.u32(at, at, length, pos);
        mad.lo.u32(at, at, half, pair);
        load(co, rope, at);
        mul.lo.u32(n, heads, length);
        mul.lo.u32(n, n, half);
        mad.lo.u32(at, n, 2, at);
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

    /// One warp per head and band of a layer: A, E and C of the far
    /// rotations `rotated` and their adjoint `dy`, at
    /// `sums[3·(head·bands + band)]` and the two after it. `shape[head]` is
    /// the head's window. Each lane sums its rows in order, then the warp
    /// reduces in a fixed tree.
    pub unsafe extern "ptx-kernel" fn torques(
        sums: *mut f32,
        rotated: *const f32,
        dy: *const f32,
        shape: *const f32,
        bands: u32,
        heads: u32,
        length: u32,
        rows: u32,
    ) {
        let (block, lane, head, band, d, width, stride, row, pos, at): (
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
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        div.u32(head, block, bands);
        rem.u32(band, block, bands);
        shl.b32(d, bands, 1);
        mul.lo.u32(width, d, heads);
        mul.lo.u32(stride, width, 3);
        let (w, position, excess, value, other, total, excessive, keyed): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        load(w, shape, head);
        mov.f32(total, 0.0);
        mov.f32(excessive, 0.0);
        mov.f32(keyed, 0.0);
        mov.u32(row, lane);
        let done: pred;
        label(next);
        setp.ge.u32(done, row, rows);
        if done {
            bra(finish);
        }
        rem.u32(pos, row, length);
        cvt.rn.f32.u32(position, pos);
        mad.lo.u32(at, row, stride, band);
        mad.lo.u32(at, head, d, at);
        torque(value, rotated, dy, at, bands);
        add.rn.f32(total, total, value);
        sub.rn.f32(excess, position, w);
        fma.rn.f32(excessive, excess, value, excessive);
        add.u32(at, at, width);
        torque(value, rotated, dy, at, bands);
        fma.rn.f32(keyed, position, value, keyed);
        add.u32(row, row, 32);
        bra(next);
        label(finish);
        for j in 0..5 {
            shfl.sync.bfly.b32(other, total, 1 << j, 31, 0xffffffff);
            add.rn.f32(total, total, other);
            shfl.sync.bfly.b32(other, excessive, 1 << j, 31, 0xffffffff);
            add.rn.f32(excessive, excessive, other);
            shfl.sync.bfly.b32(other, keyed, 1 << j, 31, 0xffffffff);
            add.rn.f32(keyed, keyed, other);
        }
        setp.eq.u32(done, lane, 0);
        if done {
            mad.lo.u32(at, head, bands, band);
            mul.lo.u32(at, at, 3);
            store(sums, at, total);
            add.u32(at, at, 1);
            store(sums, at, excessive);
            add.u32(at, at, 1);
            store(sums, at, keyed);
        }
    }

    /// One layer's gradients from its sums, in one block of at least
    /// max(bands, heads) threads. Thread `i < bands` adds the far frequency
    /// gradient θ̂ Σ_heads (wA + s(E + C)) to `alpha_gradient[i]`: the far
    /// query sits at w + s(t − w), the far key at s·j. Thread `i < heads`
    /// stores ω's gradient w(1 − s)Σ θ̂A at `window_gradient[i]` and σ's,
    /// s(1 − s)Σ θ̂(E + C), at `window_gradient[stride + i]`. `shape` holds
    /// the layer's w at `[head]`, s at `[stride + head]` and 1 − s at
    /// `[2·stride + head]`.
    pub unsafe extern "ptx-kernel" fn finish(
        alpha_gradient: *mut f32,
        window_gradient: *mut f32,
        sums: *const f32,
        shape: *const f32,
        frequencies: *const f32,
        alpha: *const f32,
        bands: u32,
        heads: u32,
        stride: u32,
    ) {
        let (i, head, band, at): (u32, u32, u32, u32);
        mov.u32(i, tid.x);
        let (frequency, w, s, rest, area, excessive, keyed, value, total, slope): (
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
            f32,
        );
        let (skip, done): (pred, pred);
        setp.ge.u32(skip, i, bands);
        if skip {
            bra(window);
        }
        learned(frequency, frequencies, alpha, i);
        mov.f32(total, 0.0);
        mov.u32(head, 0);
        label(next_head);
        setp.ge.u32(done, head, heads);
        if done {
            bra(frequency_done);
        }
        load(w, shape, head);
        add.u32(at, head, stride);
        load(s, shape, at);
        mad.lo.u32(at, head, bands, i);
        mul.lo.u32(at, at, 3);
        load(area, sums, at);
        add.u32(at, at, 1);
        load(excessive, sums, at);
        add.u32(at, at, 1);
        load(keyed, sums, at);
        add.rn.f32(keyed, excessive, keyed);
        mul.rn.f32(value, w, area);
        fma.rn.f32(value, s, keyed, value);
        add.rn.f32(total, total, value);
        add.u32(head, head, 1);
        bra(next_head);
        label(frequency_done);
        mul.rn.f32(total, frequency, total);
        load(value, alpha_gradient, i);
        add.rn.f32(value, value, total);
        store(alpha_gradient, i, value);
        label(window);
        setp.ge.u32(skip, i, heads);
        if skip {
            ret();
        }
        mov.f32(total, 0.0);
        mov.f32(slope, 0.0);
        mov.u32(band, 0);
        label(next_band);
        setp.ge.u32(done, band, bands);
        if done {
            bra(window_done);
        }
        learned(frequency, frequencies, alpha, band);
        mad.lo.u32(at, i, bands, band);
        mul.lo.u32(at, at, 3);
        load(area, sums, at);
        add.u32(at, at, 1);
        load(excessive, sums, at);
        add.u32(at, at, 1);
        load(keyed, sums, at);
        fma.rn.f32(total, frequency, area, total);
        add.rn.f32(keyed, excessive, keyed);
        fma.rn.f32(slope, frequency, keyed, slope);
        add.u32(band, band, 1);
        bra(next_band);
        label(window_done);
        load(w, shape, i);
        add.u32(at, i, stride);
        load(s, shape, at);
        add.u32(at, at, stride);
        load(rest, shape, at);
        mul.rn.f32(value, w, rest);
        mul.rn.f32(value, value, total);
        store(window_gradient, i, value);
        mul.rn.f32(value, s, rest);
        mul.rn.f32(value, value, slope);
        add.u32(at, i, stride);
        store(window_gradient, at, value);
    }
}

/// LeRoPE's base frequencies 10000^(−2·band/D) in the host FP32 arithmetic
/// of gpt's `rotary::base_tables`: its angle at position 1.
pub fn frequencies(head_dim: usize) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|band| 1. / 10000f32.powf(2. * band as f32 / head_dim as f32))
        .collect()
}

//! ReRoPE's attention adjoint. Four kernels of gpt 5a37c51, each with one
//! change: `ordinary_attention::query_backward`, `attention::key_backward`,
//! `phase_score::query_correction` and `phase_score::key_correction`. The
//! forward (`rerope_forward`) scored the pair of the query `t` and the key
//! `key` from the near rotations `qkv` when `t - key < window` and from the
//! far rotations `far` beyond it, so the pair's gradient reads the operands of
//! its region and goes to the adjoint of that region: `dqkv` near, `dfar` far.
//! Each kernel keeps one accumulator per region; the near one is updated by
//! the control's operations in the control's order, and a select keeps it
//! unchanged on a far pair. The values do not depend on the position, so
//! their adjoint is the control's. The load and store helpers are the
//! control's (`ordinary_attention` wraps the same load in a register copy).
//!
//! Experiment copy of `experiments/rerope/src/rerope_backward.rs`. One
//! change: the window is the head's, `windows[h]`, so that each head of a
//! layer may learn its own (`window_rotary`). Two heads that share a warp at
//! D = 16 keep the same loop bounds; the region only selects operands.
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
    // The head's window: a pair is near when its distance is below it.
    fn load_window(v: u32, p: u64, i: u32) {
        let a: u64;
        mad.wide.u32(a, i, 4, p);
        ld.global.u32(v, [a]);
    }
    fn sum<const STEPS: usize>(v: f32) {
        let x: f32;
        for j in 0..STEPS {
            shfl.sync.bfly.b32(x, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, x);
        }
    }
    fn head_index(id: u32, lane: u32) {
        let warp: u32;
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
    }
    // d Im(conj(q)*k)/dq = (k_im,-k_re), in the within-head layout.
    fn cross<const D: usize>(v: f32, p: u64, base: u32, col: u32) {
        let (at, other): (u32, u32);
        let real: pred;
        let negative: f32;
        setp.lt.u32(real, col, D / 2);
        add.u32(at, col, D / 2);
        sub.u32(other, col, D / 2);
        selp.u32(at, at, other, real);
        add.u32(at, base, at);
        load(v, p, at);
        neg.f32(negative, v);
        selp.f32(v, v, negative, real);
    }
    /// `ordinary_attention::query_backward`: the key of a far pair is the far key.
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn query_backward<const D: usize>(
        dq: *mut f32,
        dfar: *mut f32,
        ds: *mut f32,
        gd: *mut f32,
        dv: *mut f32,
        dy: *const f32,
        prob: *const f32,
        qkv: *const f32,
        far: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
        windows: *const u32,
    ) {
        let (id, lane, b, t, h, w, base, qbase, anchor, scorebase, key, at, col, outbase): (
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
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
        if D == 16 {
            // Compile-time specialization: two adjacent heads of the same
            // query position share a warp. Their loop bounds stay identical.
            shr.u32(key, lane, 4);
            mad.lo.u32(id, id, 2, key);
            and.b32(lane, lane, 15);
        }
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
        rem.u32(h, id, heads);
        let window: u32;
        load_window(window, windows, h);
        div.u32(t, id, heads);
        div.u32(b, t, length);
        rem.u32(t, t, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mul.lo.u32(qbase, t, 3);
        mad.lo.u32(qbase, qbase, w, base);
        mad.lo.u32(qbase, h, D, qbase);
        mad.lo.u32(anchor, h, D, base);
        mad.lo.u32(anchor, w, 2, anchor);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        mad.lo.u32(outbase, b, length, t);
        mul.lo.u32(outbase, outbase, w);
        mad.lo.u32(outbase, h, D, outbase);
        let (first, gradient, dquery, dfarquery): (
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
        );
        let (s, z, p, total, value, gain, dsval, next): (f32, f32, f32, f32, f32, f32, f32, f32);
        let (distance, source): (u32, u64);
        let near: pred;
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            mov.f32(first[j], 0.0f32);
            mov.f32(gradient[j], 0.0f32);
            mov.f32(dquery[j], 0.0f32);
            mov.f32(dfarquery[j], 0.0f32);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, anchor, col);
                load(first[j], qkv, at);
                add.u32(at, outbase, col);
                load(gradient[j], dy, at);
                store(gd, at, gradient[j]);
                mov.f32(z, 0.0f32);
                store(dv, at, z);
            }
        }
        add.u32(at, t, 1);
        let gain32: f32;
        cvt.rn.f32.u32(gain32, at);
        lg2.approx.f32(gain32, gain32);
        mul.rn.f32(gain32, gain32, 0.6931471805599453 / (D as f32));
        mov.f32(gain, gain32);
        mov.f32(total, 0.0f32);
        // Two passes avoid rounding the intermediate dP dot products to FP32.
        for pass in 0..2 {
            mov.u32(key, 0);
            label(keys);
            mov.f32(s, 0.0f32);
            for j in 0..(D + 31) / 32 {
                add.u32(col, lane, 32 * j);
                let active: pred;
                setp.lt.u32(active, col, D);
                if active {
                    mul.lo.u32(at, key, 3);
                    mad.lo.u32(at, at, w, anchor);
                    add.u32(at, at, col);
                    load(value, qkv, at);
                    sub.rn.f32(value, value, first[j]);
                    mul.rn.f32(z, gradient[j], value);
                    add.rn.f32(s, s, z);
                }
            }
            if D == 16 {
                sum::<4>(s);
            } else {
                sum::<5>(s);
            }
            add.u32(at, scorebase, key);
            load(p, prob, at);
            if pass == 0 {
                mul.rn.f32(z, p, s);
                add.rn.f32(total, total, z);
            } else {
                sub.rn.f32(s, s, total);
                mul.rn.f32(s, s, p);
                mul.rn.f32(dsval, s, gain);
                let lead: pred;
                setp.eq.u32(lead, lane, 0);
                if lead {
                    store(ds, at, dsval);
                }
                // The region of the pair, as the forward chose it.
                sub.u32(distance, t, key);
                setp.lt.u32(near, distance, window);
                selp.u64(source, qkv, far, near);
                for j in 0..(D + 31) / 32 {
                    add.u32(col, lane, 32 * j);
                    let active: pred;
                    setp.lt.u32(active, col, D);
                    if active {
                        mul.lo.u32(at, key, 3);
                        add.u32(at, at, 1);
                        mad.lo.u32(at, at, w, base);
                        mad.lo.u32(at, h, D, at);
                        add.u32(at, at, col);
                        load(value, source, at);
                        mul.rn.f32(z, dsval, value);
                        add.rn.f32(next, dquery[j], z);
                        selp.f32(dquery[j], next, dquery[j], near);
                        add.rn.f32(next, dfarquery[j], z);
                        selp.f32(dfarquery[j], dfarquery[j], next, near);
                    }
                }
            }
            add.u32(key, key, 1);
            let more: pred;
            setp.le.u32(more, key, t);
            if more {
                bra(keys);
            }
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                store(dq, at, dquery[j]);
                store(dfar, at, dfarquery[j]);
            }
        }
    }
    /// `attention::key_backward`: the query of a far pair is the far query.
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn key_backward<const D: usize>(
        dqkv: *mut f32,
        dfar: *mut f32,
        ds: *const f32,
        gd: *const f32,
        dv: *const f32,
        prob: *const f32,
        qkv: *const f32,
        far: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
        windows: *const u32,
    ) {
        let (id, lane, b, key, h, w, base, outbase, scorebase, t, at, col, qbase): (
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
        mov.u32(id, ctaid.x);
        mov.u32(lane, tid.x);
        let warp: u32;
        shr.u32(warp, lane, 5);
        mad.lo.u32(id, id, 4, warp);
        and.b32(lane, lane, 31);
        if D == 16 {
            shr.u32(warp, lane, 4);
            mad.lo.u32(id, id, 2, warp);
            and.b32(lane, lane, 15);
        }
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
        rem.u32(h, id, heads);
        let window: u32;
        load_window(window, windows, h);
        div.u32(key, id, heads);
        div.u32(b, key, length);
        rem.u32(key, key, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mad.lo.u32(outbase, base, w, 0);
        mad.lo.u32(outbase, h, D, outbase);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mad.lo.u32(base, h, D, base);
        mad.lo.u32(scorebase, b, heads, h);
        mul.lo.u32(scorebase, scorebase, length);
        mul.lo.u32(scorebase, scorebase, length);
        let (dk, dfarkey, dvalue): (
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
            [f32; (D + 31) / 32],
        );
        let (s, p, z, v, next): (f32, f32, f32, f32, f32);
        let (distance, source): (u32, u64);
        let near: pred;
        for j in 0..(D + 31) / 32 {
            mov.f32(dk[j], 0.0);
            mov.f32(dfarkey[j], 0.0);
            mov.f32(dvalue[j], 0.0);
        }
        mov.u32(t, key);
        label(query);
        mad.lo.u32(at, t, length, key);
        add.u32(at, at, scorebase);
        load(s, ds, at);
        load(p, prob, at);
        mul.lo.u32(qbase, t, 3);
        mad.lo.u32(qbase, qbase, w, base);
        // The region of the pair, as the forward chose it.
        sub.u32(distance, t, key);
        setp.lt.u32(near, distance, window);
        selp.u64(source, qkv, far, near);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(v, source, at);
                mul.rn.f32(z, s, v);
                add.rn.f32(next, dk[j], z);
                selp.f32(dk[j], next, dk[j], near);
                add.rn.f32(next, dfarkey[j], z);
                selp.f32(dfarkey[j], dfarkey[j], next, near);
                mad.lo.u32(at, t, w, outbase);
                add.u32(at, at, col);
                load(v, gd, at);
                mul.rn.f32(z, p, v);
                add.rn.f32(dvalue[j], dvalue[j], z);
            }
        }
        add.u32(t, t, 1);
        let more: pred;
        setp.lt.u32(more, t, length);
        if more {
            bra(query);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                mul.lo.u32(at, key, 3);
                add.u32(at, at, 1);
                mad.lo.u32(at, at, w, base);
                add.u32(at, at, col);
                store(dqkv, at, dk[j]);
                store(dfar, at, dfarkey[j]);
                add.u32(qbase, at, w);
                mad.lo.u32(at, key, w, outbase);
                add.u32(at, at, col);
                load(v, dv, at);
                add.rn.f32(v, v, dvalue[j]);
                store(dqkv, qbase, v);
            }
        }
    }
    /// `phase_score::query_correction`: the key of a far pair is the far key.
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn query_correction<const D: usize>(
        dqkv: *mut f32,
        dfar: *mut f32,
        parts: *mut f32,
        ds: *const f32,
        imaginary: *const f32,
        qkv: *const f32,
        far: *const f32,
        coefficients: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
        windows: *const u32,
    ) {
        let (id, lane, h, b, t, w, base, qbase, scorebase, key, col, at, kbase): (
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
        head_index(id, lane);
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
        rem.u32(h, id, heads);
        let window: u32;
        load_window(window, windows, h);
        div.u32(t, id, heads);
        div.u32(b, t, length);
        rem.u32(t, t, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mul.lo.u32(qbase, t, 3);
        mad.lo.u32(qbase, qbase, w, base);
        mad.lo.u32(qbase, h, D, qbase);
        mad.lo.u32(scorebase, b, heads, h);
        mad.lo.u32(scorebase, scorebase, length, t);
        mul.lo.u32(scorebase, scorebase, length);
        let (beta, derivative, im, g, c, v, term, da, next): (
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
        let (distance, source): (u32, u64);
        let near: pred;
        mul.lo.u32(at, h, 2);
        load(beta, coefficients, at);
        add.u32(at, at, 1);
        load(derivative, coefficients, at);
        let (grad, gradfar): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        for j in 0..(D + 31) / 32 {
            mov.f32(grad[j], 0.0);
            mov.f32(gradfar[j], 0.0);
        }
        mov.f32(da, 0.0);
        mov.u32(key, 0);
        label(keys);
        add.u32(at, scorebase, key);
        load(im, imaginary, at);
        load(g, ds, at);
        mul.rn.f32(c, im, 1.0 / (D as f32));
        mul.rn.f32(term, g, c);
        mul.rn.f32(term, term, im);
        mul.rn.f32(term, term, derivative);
        add.rn.f32(da, da, term);
        mul.rn.f32(c, c, beta);
        mul.rn.f32(c, c, 2.0);
        mul.rn.f32(c, c, g);
        mul.lo.u32(kbase, key, 3);
        add.u32(kbase, kbase, 1);
        mad.lo.u32(kbase, kbase, w, base);
        mad.lo.u32(kbase, h, D, kbase);
        // The region of the pair, as the forward chose it.
        sub.u32(distance, t, key);
        setp.lt.u32(near, distance, window);
        selp.u64(source, qkv, far, near);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                cross::<D>(v, source, kbase, col);
                mul.rn.f32(term, c, v);
                add.rn.f32(next, grad[j], term);
                selp.f32(grad[j], next, grad[j], near);
                add.rn.f32(next, gradfar[j], term);
                selp.f32(gradfar[j], gradfar[j], next, near);
            }
        }
        add.u32(key, key, 1);
        let more: pred;
        setp.le.u32(more, key, t);
        if more {
            bra(keys);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, qbase, col);
                load(v, dqkv, at);
                add.rn.f32(v, v, grad[j]);
                store(dqkv, at, v);
                load(v, dfar, at);
                add.rn.f32(v, v, gradfar[j]);
                store(dfar, at, v);
            }
        }
        let lead: pred;
        setp.eq.u32(lead, lane, 0);
        if lead {
            store(parts, id, da);
        }
    }
    /// `phase_score::key_correction`: the query of a far pair is the far query.
    #[instances(16, 32, 128)]
    pub unsafe extern "ptx-kernel" fn key_correction<const D: usize>(
        dqkv: *mut f32,
        dfar: *mut f32,
        ds: *const f32,
        imaginary: *const f32,
        qkv: *const f32,
        far: *const f32,
        coefficients: *const f32,
        heads: u32,
        length: u32,
        head_count: u32,
        windows: *const u32,
    ) {
        let (id, lane, h, b, key, w, base, kbase, scorebase, query, col, at, qbase): (
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
        head_index(id, lane);
        let past: pred;
        setp.ge.u32(past, id, head_count);
        if past {
            ret();
        }
        rem.u32(h, id, heads);
        let window: u32;
        load_window(window, windows, h);
        div.u32(key, id, heads);
        div.u32(b, key, length);
        rem.u32(key, key, length);
        mul.lo.u32(w, heads, D);
        mul.lo.u32(base, b, length);
        mul.lo.u32(base, base, 3);
        mul.lo.u32(base, base, w);
        mul.lo.u32(kbase, key, 3);
        add.u32(kbase, kbase, 1);
        mad.lo.u32(kbase, kbase, w, base);
        mad.lo.u32(kbase, h, D, kbase);
        mad.lo.u32(scorebase, b, heads, h);
        mul.lo.u32(scorebase, scorebase, length);
        let (beta, im, g, c, v, term, next): (f32, f32, f32, f32, f32, f32, f32);
        let (distance, source): (u32, u64);
        let near: pred;
        mul.lo.u32(at, h, 2);
        load(beta, coefficients, at);
        let (grad, gradfar): ([f32; (D + 31) / 32], [f32; (D + 31) / 32]);
        for j in 0..(D + 31) / 32 {
            mov.f32(grad[j], 0.0);
            mov.f32(gradfar[j], 0.0);
        }
        mov.u32(query, key);
        label(queries);
        add.u32(at, scorebase, query);
        mad.lo.u32(at, at, length, key);
        load(im, imaginary, at);
        load(g, ds, at);
        mul.rn.f32(c, im, 1.0 / (D as f32));
        mul.rn.f32(c, c, beta);
        mul.rn.f32(c, c, -2.0);
        mul.rn.f32(c, c, g);
        mul.lo.u32(qbase, query, 3);
        mad.lo.u32(qbase, qbase, w, base);
        mad.lo.u32(qbase, h, D, qbase);
        // The region of the pair, as the forward chose it.
        sub.u32(distance, query, key);
        setp.lt.u32(near, distance, window);
        selp.u64(source, qkv, far, near);
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                cross::<D>(v, source, qbase, col);
                mul.rn.f32(term, c, v);
                add.rn.f32(next, grad[j], term);
                selp.f32(grad[j], next, grad[j], near);
                add.rn.f32(next, gradfar[j], term);
                selp.f32(gradfar[j], gradfar[j], next, near);
            }
        }
        add.u32(query, query, 1);
        let more: pred;
        setp.lt.u32(more, query, length);
        if more {
            bra(queries);
        }
        for j in 0..(D + 31) / 32 {
            add.u32(col, lane, 32 * j);
            let active: pred;
            setp.lt.u32(active, col, D);
            if active {
                add.u32(at, kbase, col);
                load(v, dqkv, at);
                add.rn.f32(v, v, grad[j]);
                store(dqkv, at, v);
                load(v, dfar, at);
                add.rn.f32(v, v, gradfar[j]);
                store(dfar, at, v);
            }
        }
    }
}

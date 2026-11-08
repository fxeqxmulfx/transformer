//! FP64 independent A norms and one rounded owner update. The signed map
//! is the original model::pairs contract; E/B use the source AdamW kernels.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn load_u(x: u32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.u32(x, [p]);
    }
    fn load_d(x: f64, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 8, base);
        ld.global.f64(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn store_d(base: u64, i: u32, x: f64) {
        let p: u64;
        mad.wide.u32(p, i, 8, base);
        st.global.f64([p], x);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn sum(x: f64) {
        let (lo, hi, ol, oh): (u32, u32, u32, u32);
        let v: f64;
        for j in 0..5 {
            mov.b64((lo, hi), x);
            shfl.sync.bfly.b32(ol, lo, 1 << j, 31, 0xffffffff);
            shfl.sync.bfly.b32(oh, hi, 1 << j, 31, 0xffffffff);
            mov.b64(v, (ol, oh));
            add.rn.f64(x, x, v);
        }
    }
    fn addresses(ar: u32, bi: u32, br: u32, ai: u32, map: u64, i: u32, mode: u32) {
        let complex: pred;
        setp.ne.u32(complex, mode, 0);
        add.u32(ar, i, 12416);
        mov.u32(bi, ar);
        mov.u32(br, ar);
        mov.u32(ai, ar);
        if complex {
            let at: u32;
            mul.lo.u32(at, i, 5);
            load_u(ar, map, at);
            add.u32(at, at, 1);
            load_u(bi, map, at);
            add.u32(at, at, 1);
            load_u(br, map, at);
            add.u32(at, at, 1);
            load_u(ai, map, at);
        }
    }
    fn values(pa: f64, pb: f64, ga: f64, gb: f64, p: u64, g: u64, map: u64, i: u32, mode: u32) {
        let (ar, bi, br, ai): (u32, u32, u32, u32);
        addresses(ar, bi, br, ai, map, i, mode);
        let z: f32;
        let t: f64;
        load(z, p, ar);
        cvt.f64.f32(pa, z);
        load(z, g, ar);
        cvt.f64.f32(ga, z);
        mov.f64(pb, 0.0f64);
        mov.f64(gb, 0.0f64);
        let complex: pred;
        setp.ne.u32(complex, mode, 0);
        if complex {
            load(z, p, bi);
            cvt.f64.f32(pb, z);
            load(z, g, ai);
            cvt.f64.f32(t, z);
            add.rn.f64(ga, ga, t);
            load(z, g, bi);
            cvt.f64.f32(gb, z);
            load(z, g, br);
            cvt.f64.f32(t, z);
            sub.rn.f64(gb, gb, t);
        }
    }
    pub unsafe extern "ptx-kernel" fn energies(
        out: *mut f64,
        p: *const f32,
        g: *const f32,
        map: *const u32,
        count: u32,
        mode: u32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (pa, pb, ga, gb, ps, gs, t): (f64, f64, f64, f64, f64, f64, f64);
        values(pa, pb, ga, gb, p, g, map, i, mode);
        mul.rn.f64(ps, pa, pa);
        mul.rn.f64(t, pb, pb);
        add.rn.f64(ps, ps, t);
        mul.rn.f64(gs, ga, ga);
        mul.rn.f64(t, gb, gb);
        add.rn.f64(gs, gs, t);
        sum(ps);
        sum(gs);
        let (lane, at): (u32, u32);
        and.b32(lane, i, 31);
        setp.eq.u32(past, lane, 0);
        if past {
            shr.u32(at, i, 5);
            mul.lo.u32(at, at, 2);
            store_d(out, at, ps);
            add.u32(at, at, 1);
            store_d(out, at, gs);
        }
    }
    pub unsafe extern "ptx-kernel" fn factor(
        out: *mut f64,
        partial: *const f64,
        parts: u32,
        alpha: f64,
    ) {
        let lane: u32;
        mov.u32(lane, tid.x);
        let (ps, gs, pv, gv, t, scale, zeros): (f64, f64, f64, f64, f64, f64, f64);
        mov.f64(ps, 0.0f64);
        mov.f64(gs, 0.0f64);
        let (k, at): (u32, u32);
        let valid: pred;
        for j in 0..86 {
            add.u32(k, lane, 32 * j);
            setp.lt.u32(valid, k, parts);
            if valid {
                mul.lo.u32(at, k, 2);
                load_d(pv, partial, at);
                add.u32(at, at, 1);
                load_d(gv, partial, at);
                add.rn.f64(ps, ps, pv);
                add.rn.f64(gs, gs, gv);
            }
        }
        sum(ps);
        sum(gs);
        setp.ne.u32(valid, lane, 0);
        if valid {
            ret();
        }
        let (zero_p, zero_g, zero): (pred, pred, pred);
        setp.eq.f64(zero_p, ps, 0.0f64);
        setp.eq.f64(zero_g, gs, 0.0f64);
        or.pred(zero, zero_p, zero_g);
        mov.u32(at, 3);
        load_d(zeros, out, at);
        if zero {
            mov.f64(scale, 0.0f64);
            add.rn.f64(zeros, zeros, 1.0f64);
        } else {
            sqrt.rn.f64(t, ps);
            mul.rn.f64(t, t, alpha);
            sqrt.rn.f64(scale, gs);
            div.rn.f64(scale, t, scale);
        }
        mov.u32(at, 0);
        store_d(out, at, ps);
        add.u32(at, at, 1);
        store_d(out, at, gs);
        add.u32(at, at, 1);
        store_d(out, at, scale);
        add.u32(at, at, 1);
        store_d(out, at, zeros);
    }
    pub unsafe extern "ptx-kernel" fn apply(
        p: *mut f32,
        g: *const f32,
        map: *const u32,
        stats: *const f64,
        count: u32,
        mode: u32,
    ) {
        let i: u32;
        index(i);
        let stop: pred;
        setp.ge.u32(stop, i, count);
        if stop {
            ret();
        }
        let scale: f64;
        let at: u32;
        mov.u32(at, 2);
        load_d(scale, stats, at);
        setp.eq.f64(stop, scale, 0.0f64);
        if stop {
            ret();
        }
        let (pa, pb, ga, gb, t): (f64, f64, f64, f64, f64);
        values(pa, pb, ga, gb, p, g, map, i, mode);
        let (ar, bi, br, ai): (u32, u32, u32, u32);
        addresses(ar, bi, br, ai, map, i, mode);
        let z: f32;
        mul.rn.f64(t, ga, scale);
        sub.rn.f64(pa, pa, t);
        cvt.rn.f32.f64(z, pa);
        store(p, ar, z);
        setp.ne.u32(stop, mode, 0);
        if stop {
            store(p, ai, z);
            mul.rn.f64(t, gb, scale);
            sub.rn.f64(pb, pb, t);
            cvt.rn.f32.f64(z, pb);
            store(p, bi, z);
            neg.f32(z, z);
            store(p, br, z);
        }
    }
}

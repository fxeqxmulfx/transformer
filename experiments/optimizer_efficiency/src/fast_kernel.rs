//! Exact source AdamW arithmetic (pinned rotate90_complex infrastructure/ops.rs
//! and infrastructure/product.rs,
//! SHA 14cff3b5), with uniform scalars precomputed and compact moment owners.
//! No reassociation of variance normalization or epsilon placement.
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
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn index(i: u32) {
        let (b, t, n): (u32, u32, u32);
        mov.u32(b, ctaid.x);
        mov.u32(t, tid.x);
        mov.u32(n, ntid.x);
        mad.lo.u32(i, b, n, t);
    }
    fn moment(p: f32, m: f32, v: f32, g: f32, square: f32, ratio: f32, bc2: f32, decay: f32) {
        let (z, t): (f32, f32);
        sub.rn.f32(z, g, m);
        fma.rn.f32(m, 0.1, z, m);
        mul.rn.f32(v, v, 0.98);
        fma.rn.f32(v, 0.02, square, v);
        sqrt.rn.f32(z, v);
        div.rn.f32(z, z, bc2);
        add.rn.f32(z, z, 1e-8);
        mul.rn.f32(t, ratio, m);
        div.rn.f32(t, t, z);
        mul.rn.f32(p, p, decay);
        sub.rn.f32(p, p, t);
    }
    pub unsafe extern "ptx-kernel" fn dense(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        count: u32,
        ratio: f32,
        bc2: f32,
        decay: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        let (pi, mi, vi, gi, square): (f32, f32, f32, f32, f32);
        load(pi, p, i);
        load(mi, m, i);
        load(vi, v, i);
        load(gi, g, i);
        mul.rn.f32(square, gi, gi);
        moment(pi, mi, vi, gi, square, ratio, bc2, decay);
        store(p, i, pi);
        store(m, i, mi);
        store(v, i, vi);
    }
    pub unsafe extern "ptx-kernel" fn complex(
        p: *mut f32,
        m: *mut f32,
        v: *mut f32,
        g: *const f32,
        map: *const u32,
        count: u32,
        ratio: f32,
        bc2: f32,
        decay: f32,
    ) {
        let (i, at, ar, bi, br, ai, kind, ma, mb): (u32, u32, u32, u32, u32, u32, u32, u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, count);
        if past {
            ret();
        }
        mul.lo.u32(at, i, 5);
        load_u(ar, map, at);
        add.u32(at, at, 1);
        load_u(bi, map, at);
        add.u32(at, at, 1);
        load_u(br, map, at);
        add.u32(at, at, 1);
        load_u(ai, map, at);
        add.u32(at, at, 1);
        load_u(kind, map, at);
        mul.lo.u32(ma, i, 2);
        add.u32(mb, ma, 1);
        let (ga, gb, pa, pb, mia, mib, va, vb, sa, sb, t): (
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
        load(ga, g, ar);
        load(gb, g, bi);
        let matrix: pred;
        setp.eq.u32(matrix, kind, 0);
        if matrix {
            load(t, g, ai);
            add.rn.f32(ga, ga, t);
            load(t, g, br);
            sub.rn.f32(gb, gb, t);
        }
        mul.rn.f32(sa, ga, ga);
        mul.rn.f32(sb, gb, gb);
        load(pa, p, ar);
        load(pb, p, bi);
        load(mia, m, ma);
        load(mib, m, mb);
        load(va, v, ma);
        load(vb, v, mb);
        moment(pa, mia, va, ga, sa, ratio, bc2, decay);
        moment(pb, mib, vb, gb, sb, ratio, bc2, decay);
        store(p, ar, pa);
        store(p, bi, pb);
        store(m, ma, mia);
        store(m, mb, mib);
        store(v, ma, va);
        store(v, mb, vb);
        if matrix {
            neg.f32(pb, pb);
            store(p, ai, pa);
            store(p, br, pb);
        }
    }
}

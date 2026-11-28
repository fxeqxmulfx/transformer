//! Packing, normalization and updates around the decoder's batched complex GEMM.
#[kernite_macros::kernels]
pub(super) mod device {
    fn index(i: u32) {
        let (block, lane, threads): (u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(threads, ntid.x);
        mad.lo.u32(i, block, threads, lane);
    }
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn put(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }
    fn mapped(j: u32, map: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, map);
        ld.global.u32(j, [p]);
    }

    pub unsafe extern "ptx-kernel" fn momentum(
        x: *mut f32, moment: *mut f32, grad: *const f32,
        map: *const u32, norm: *const f32, n: u32,
    ) {
        let (i, j): (u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past { ret(); }
        mapped(j, map, i);
        let (g, v, s): (f32, f32, f32);
        ld.global.f32(s, [norm]);
        max.f32(s, s, 1.0);
        load(g, grad, j);
        div.rn.f32(g, g, s);
        load(v, moment, i);
        fma.rn.f32(v, v, 0.95, g);
        put(moment, i, v);
        fma.rn.f32(v, v, 0.95, g);
        put(x, i, v);
    }

    // One warp per matrix. Real and imaginary squares both enter the Frobenius norm.
    pub unsafe extern "ptx-kernel" fn normalize(x: *mut f32, width: u32) {
        let (block, lane, start, i, j): (u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(lane, tid.x);
        mul.lo.u32(start, block, width);
        mov.u32(i, lane);
        let (s, v, peer): (f32, f32, f32);
        mov.f32(s, 0f32);
        let past: pred;
        label(sum);
        setp.ge.u32(past, i, width);
        if past { bra(reduce); }
        add.u32(j, start, i);
        load(v, x, j);
        fma.rn.f32(s, v, v, s);
        add.u32(i, i, 32);
        bra(sum);
        label(reduce);
        for d in 0..5 {
            shfl.sync.bfly.b32(peer, s, 1 << d, 31, 0xffffffff);
            add.rn.f32(s, s, peer);
        }
        sqrt.rn.f32(s, s);
        max.f32(s, s, 1e-8);
        mov.u32(i, lane);
        label(write);
        setp.ge.u32(past, i, width);
        if past { ret(); }
        add.u32(j, start, i);
        load(v, x, j);
        div.rn.f32(v, v, s);
        put(x, j, v);
        add.u32(i, i, 32);
        bra(write);
    }

    pub unsafe extern "ptx-kernel" fn combine(
        dst: *mut f32, src: *const f32, n: u32, a: f32, b: f32,
    ) {
        let i: u32;
        index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past { ret(); }
        let (x, y): (f32, f32);
        load(x, dst, i);
        load(y, src, i);
        mul.rn.f32(x, x, a);
        fma.rn.f32(x, y, b, x);
        put(dst, i, x);
    }

    pub unsafe extern "ptx-kernel" fn apply_matrix(
        w: *mut f32, x: *const f32, map: *const u32, n: u32, shrink: f32, rate: f32,
    ) {
        let (i, j): (u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past { ret(); }
        mapped(j, map, i);
        let (v, d): (f32, f32);
        load(v, w, j);
        load(d, x, i);
        mul.rn.f32(v, v, shrink);
        mul.rn.f32(d, d, rate);
        sub.rn.f32(v, v, d);
        put(w, j, v);
    }

    pub unsafe extern "ptx-kernel" fn auxiliary(
        w: *mut f32, m: *mut f32, v: *mut f32, grad: *const f32, map: *const u32,
        norm: *const f32, n: u32, rate: f32, shrink: f32, bc1: f32, bc2: f32,
    ) {
        let (i, j): (u32, u32);
        index(i);
        let past: pred;
        setp.ge.u32(past, i, n);
        if past { ret(); }
        mapped(j, map, i);
        let (g, old, first, second, t, s): (f32, f32, f32, f32, f32, f32);
        ld.global.f32(s, [norm]);
        max.f32(s, s, 1.0);
        load(g, grad, j);
        div.rn.f32(g, g, s);
        load(first, m, i);
        mul.rn.f32(t, g, 0.1);
        mul.rn.f32(first, first, 0.9);
        add.rn.f32(first, first, t);
        load(second, v, i);
        mul.rn.f32(t, g, g);
        mul.rn.f32(t, t, 0.05);
        mul.rn.f32(second, second, 0.95);
        add.rn.f32(second, second, t);
        put(m, i, first);
        put(v, i, second);
        sqrt.rn.f32(s, second);
        div.rn.f32(s, s, bc2);
        add.rn.f32(s, s, 1e-8);
        div.rn.f32(t, rate, bc1);
        mul.rn.f32(t, t, first);
        div.rn.f32(t, t, s);
        load(old, w, j);
        mul.rn.f32(old, old, shrink);
        sub.rn.f32(old, old, t);
        put(w, j, old);
    }
}

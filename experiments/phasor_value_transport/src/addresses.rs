//! Translation acts on value addresses. The adjoint returns a gradient to the
//! first-token attention scores and clears the unused second-token gradient.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, ptr: u64, i: u32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, ptr);
        ld.global.f32(x, [addr]);
    }
    fn store(ptr: u64, i: u32, x: f32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, ptr);
        st.global.f32([addr], x);
    }
    fn argument(row: u32, offset: u32, tokens: u64, p: u32, route: u32, batch: u32) {
        let (block, thread, size, a, b): (u32, u32, u32, u32, u32);
        mov.u32(block, ctaid.x);
        mov.u32(thread, tid.x);
        mov.u32(size, ntid.x);
        mad.lo.u32(row, block, size, thread);
        let past: pred;
        setp.ge.u32(past, row, batch);
        if past {
            ret();
        }
        let addr: u64;
        mad.wide.u32(addr, row, 8, tokens);
        ld.global.u32(a, [addr]);
        add.u64(addr, addr, 4);
        ld.global.u32(b, [addr]);
        mul.lo.u32(offset, a, b);
        mul.lo.u32(offset, offset, route);
        rem.u32(offset, offset, p);
        mul.lo.u32(row, row, 2);
        mul.lo.u32(row, row, p);
    }
    fn translated(at: u32, row: u32, s: u32, offset: u32, p: u32) {
        add.u32(at, s, offset);
        rem.u32(at, at, p);
        add.u32(at, at, p);
        add.u32(at, at, row);
    }
    pub unsafe extern "ptx-kernel" fn forward(
        weights: *mut f32,
        scores: *const f32,
        tokens: *const u32,
        p: u32,
        route: u32,
        batch: u32,
    ) {
        let (row, offset, s, at, dest): (u32, u32, u32, u32, u32);
        argument(row, offset, tokens, p, route, batch);
        let (top, value, mass, zero): (f32, f32, f32, f32);
        mov.f32(zero, 0.0);
        let more: pred;
        mov.f32(top, f32::NEG_INFINITY);
        mov.u32(s, 0);
        label(maximum);
        add.u32(at, row, s);
        load(value, scores, at);
        max.f32(top, top, value);
        add.u32(s, s, 1);
        setp.lt.u32(more, s, p);
        if more {
            bra(maximum);
        }
        mov.f32(mass, 0.0);
        mov.u32(s, 0);
        label(collect);
        add.u32(at, row, s);
        load(value, scores, at);
        sub.rn.f32(value, value, top);
        mul.rn.f32(value, value, 1.4426950408889634);
        ex2.approx.f32(value, value);
        add.rn.f32(mass, mass, value);
        translated(dest, row, s, offset, p);
        store(weights, dest, value);
        store(weights, at, zero);
        add.u32(s, s, 1);
        setp.lt.u32(more, s, p);
        if more {
            bra(collect);
        }
        mov.u32(s, 0);
        label(normalize);
        add.u32(at, row, p);
        add.u32(at, at, s);
        load(value, weights, at);
        div.rn.f32(value, value, mass);
        store(weights, at, value);
        add.u32(s, s, 1);
        setp.lt.u32(more, s, p);
        if more {
            bra(normalize);
        }
    }
    pub unsafe extern "ptx-kernel" fn backward(
        dscores: *mut f32,
        weights: *const f32,
        dweights: *const f32,
        tokens: *const u32,
        p: u32,
        route: u32,
        batch: u32,
    ) {
        let (row, offset, s, at, dest): (u32, u32, u32, u32, u32);
        argument(row, offset, tokens, p, route, batch);
        let (w, derivative, mean, zero): (f32, f32, f32, f32);
        mov.f32(zero, 0.0);
        let more: pred;
        mov.f32(mean, 0.0);
        mov.u32(s, 0);
        label(expectation);
        add.u32(at, row, p);
        add.u32(at, at, s);
        load(w, weights, at);
        load(derivative, dweights, at);
        fma.rn.f32(mean, w, derivative, mean);
        add.u32(s, s, 1);
        setp.lt.u32(more, s, p);
        if more {
            bra(expectation);
        }
        mov.u32(s, 0);
        label(pullback);
        translated(dest, row, s, offset, p);
        load(w, weights, dest);
        load(derivative, dweights, dest);
        sub.rn.f32(derivative, derivative, mean);
        mul.rn.f32(derivative, derivative, w);
        add.u32(at, row, s);
        store(dscores, at, derivative);
        add.u32(at, at, p);
        store(dscores, at, zero);
        add.u32(s, s, 1);
        setp.lt.u32(more, s, p);
        if more {
            bra(pullback);
        }
    }
}

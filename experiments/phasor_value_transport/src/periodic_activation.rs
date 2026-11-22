//! PeriodicPhaseActivation.lean. One trainable strength and center per
//! channel transform the learned increment before either phase scan.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, i: u32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, i: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, i, 4, base);
        st.global.f32([p], x);
    }

    pub unsafe extern "ptx-kernel" fn forward(
        increments: *mut f32,
        raw: *mut f32,
        weights: *const f32,
    ) {
        let (row, k, at, next): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(k, tid.x);
        mad.lo.u32(at, row, 32, k);
        add.u32(next, k, 32);
        let (x, g, b, angle, sine, value): (f32, f32, f32, f32, f32, f32);
        load(x, increments, at);
        store(raw, at, x);
        load(g, weights, k);
        load(b, weights, next);
        sub.rn.f32(angle, x, b);
        add.rn.f32(angle, angle, angle);
        sin.approx.f32(sine, angle);
        mul.rn.f32(g, g, -0.5);
        fma.rn.f32(value, g, sine, x);
        store(increments, at, value);
    }

    // One warp per channel. Each projected gradient is replaced by its
    // input derivative, then the warp sums the two shared-weight gradients.
    pub unsafe extern "ptx-kernel" fn backward(
        gradients: *mut f32,
        parameters_grad: *mut f32,
        raw: *const f32,
        weights: *const f32,
        rows: u32,
    ) {
        let (row, lane, k, at, next): (u32, u32, u32, u32, u32);
        mov.u32(k, ctaid.x);
        mov.u32(lane, tid.x);
        mov.u32(row, lane);
        add.u32(next, k, 32);
        let (g, b, x, v, angle, sine, cosine, dg, db, contribution, out): (
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
        load(g, weights, k);
        load(b, weights, next);
        mov.f32(dg, 0.0);
        mov.f32(db, 0.0);
        let done: pred;
        label(scan);
        setp.ge.u32(done, row, rows);
        if done {
            bra(reduce);
        }
        mad.lo.u32(at, row, 32, k);
        load(x, raw, at);
        load(v, gradients, at);
        sub.rn.f32(angle, x, b);
        add.rn.f32(angle, angle, angle);
        sin.approx.f32(sine, angle);
        cos.approx.f32(cosine, angle);
        mul.rn.f32(sine, sine, -0.5);
        fma.rn.f32(dg, sine, v, dg);
        mul.rn.f32(contribution, g, cosine);
        fma.rn.f32(db, contribution, v, db);
        neg.f32(contribution, contribution);
        fma.rn.f32(out, contribution, v, v);
        store(gradients, at, out);
        add.u32(row, row, 32);
        bra(scan);
        label(reduce);
        for i in 0..5 {
            shfl.sync.bfly.b32(x, dg, 1 << i, 31, 0xffffffff);
            add.rn.f32(dg, dg, x);
            shfl.sync.bfly.b32(x, db, 1 << i, 31, 0xffffffff);
            add.rn.f32(db, db, x);
        }
        setp.ne.u32(done, lane, 0);
        if done {
            ret();
        }
        store(parameters_grad, k, dg);
        store(parameters_grad, next, db);
    }
}

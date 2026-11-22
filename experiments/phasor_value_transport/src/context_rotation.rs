//! Rotate the two halves of the contextual hidden vector by the exclusive
//! prefix of learned angles. One lane owns each sequence and coordinate pair.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, base: u64, idx: u32) {
        let p: u64;
        mad.wide.u32(p, idx, 4, base);
        ld.global.f32(x, [p]);
    }
    fn store(base: u64, idx: u32, x: f32) {
        let p: u64;
        mad.wide.u32(p, idx, 4, base);
        st.global.f32([p], x);
    }

    pub unsafe extern "ptx-kernel" fn forward(
        rotated: *mut f32,
        angles: *mut f32,
        context: *const f32,
        width: u32,
        length: u32,
    ) {
        let (row, end, pairs, lane, k, lo, hi): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mul.lo.u32(row, row, length);
        add.u32(end, row, length);
        shr.u32(pairs, width, 1);
        mov.u32(lane, tid.x);
        let (a, step, u, v, s, c, real, imag): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(a, 0.0);
        label(scan);
        mad.lo.u32(k, row, pairs, lane);
        mad.lo.u32(lo, row, width, lane);
        add.u32(hi, lo, pairs);
        load(step, angles, k);
        store(angles, k, a);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        load(u, context, lo);
        load(v, context, hi);
        mul.rn.f32(real, s, v);
        neg.f32(real, real);
        fma.rn.f32(real, c, u, real);
        mul.rn.f32(imag, c, v);
        fma.rn.f32(imag, s, u, imag);
        store(rotated, lo, real);
        store(rotated, hi, imag);
        add.rn.f32(a, a, step);
        add.u32(row, row, 1);
        let more: pred;
        setp.lt.u32(more, row, end);
        if more {
            bra(scan);
        }
    }

    // The cotangent of the rotated vector is replaced in place by R(-a)g.
    // Store the suffix sum before adding this row's angle derivative.
    pub unsafe extern "ptx-kernel" fn backward(
        direct: *mut f32,
        dsteps: *mut f32,
        rotated: *const f32,
        angles: *const f32,
        width: u32,
        length: u32,
    ) {
        let (row, start, pairs, lane, k, lo, hi): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(start, ctaid.x);
        mul.lo.u32(start, start, length);
        add.u32(row, start, length);
        shr.u32(pairs, width, 1);
        mov.u32(lane, tid.x);
        let (a, x, y, gx, gy, s, c, dx, dy, torque, later): (
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
        mov.f32(later, 0.0);
        label(scan);
        sub.u32(row, row, 1);
        mad.lo.u32(k, row, pairs, lane);
        mad.lo.u32(lo, row, width, lane);
        add.u32(hi, lo, pairs);
        load(a, angles, k);
        sin.approx.f32(s, a);
        cos.approx.f32(c, a);
        load(gx, direct, lo);
        load(gy, direct, hi);
        load(x, rotated, lo);
        load(y, rotated, hi);
        mul.rn.f32(dx, s, gy);
        fma.rn.f32(dx, c, gx, dx);
        mul.rn.f32(dy, s, gx);
        neg.f32(dy, dy);
        fma.rn.f32(dy, c, gy, dy);
        store(direct, lo, dx);
        store(direct, hi, dy);
        store(dsteps, k, later);
        mul.rn.f32(torque, y, gx);
        neg.f32(torque, torque);
        fma.rn.f32(torque, x, gy, torque);
        add.rn.f32(later, later, torque);
        let more: pred;
        setp.gt.u32(more, row, start);
        if more {
            bra(scan);
        }
    }
}

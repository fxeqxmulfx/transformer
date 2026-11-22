//! Learned diagonal-scaled Toeplitz operators on complex feature channels.
//! Every row reads its own value and the preceding row's controls in its sequence.
pub fn parameter_count(width: usize) -> usize {
    // Stream read, value and output projections, real stencil, complex gain and bias.
    4 * width + (10 * width - 1) * width
}

#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, p: u64, i: u32) {
        let address: u64;
        mad.wide.u32(address, i, 4, p);
        ld.global.f32(x, [address]);
    }
    fn store(p: u64, i: u32, x: f32) {
        let address: u64;
        mad.wide.u32(address, i, 4, p);
        st.global.f32([address], x);
    }

    // One thread per output channel. The identity diagonal is implicit.
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "ptx-kernel" fn forward(
        read: *mut f32,
        mixed: *mut f32,
        value: *const f32,
        kernel: *const f32,
        gain: *const f32,
        bias: *const f32,
        width: u32,
        length: u32,
    ) {
        let (r, j, stride, lo, hi, t, at, other, k, off, prefix, bank): (
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
        mov.u32(r, ctaid.x);
        mov.u32(j, tid.x);
        shl.b32(stride, width, 1);
        mad.lo.u32(lo, r, stride, j);
        add.u32(hi, lo, width);
        let (u, v, x, y, weight, gr, gi, br, bi, re, im): (
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
        load(u, value, lo);
        load(v, value, hi);
        rem.u32(t, r, length);
        mov.f32(re, u);
        mov.f32(im, v);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            bra(write);
        }
        sub.u32(prefix, r, 1);
        sub.u32(bank, stride, 1);
        mul.lo.u32(off, prefix, bank);
        add.u32(off, off, width);
        sub.u32(off, off, 1);
        sub.u32(off, off, j);
        mul.lo.u32(at, r, stride);
        add.u32(other, at, width);
        mov.u32(k, 0);
        label(mix);
        load(weight, kernel, off);
        load(x, value, at);
        load(y, value, other);
        fma.rn.f32(u, weight, x, u);
        fma.rn.f32(v, weight, y, v);
        add.u32(at, at, 1);
        add.u32(other, other, 1);
        add.u32(off, off, 1);
        add.u32(k, k, 1);
        let more: pred;
        setp.lt.u32(more, k, width);
        if more {
            bra(mix);
        }
        mad.lo.u32(at, prefix, stride, j);
        add.u32(other, at, width);
        load(gr, gain, at);
        add.rn.f32(gr, gr, 1.0);
        load(gi, gain, other);
        load(br, bias, at);
        load(bi, bias, other);
        fma.rn.f32(re, gr, u, br);
        neg.f32(x, gi);
        fma.rn.f32(re, x, v, re);
        fma.rn.f32(im, gr, v, bi);
        fma.rn.f32(im, gi, u, im);
        label(write);
        store(mixed, lo, u);
        store(mixed, hi, v);
        store(read, lo, re);
        store(read, hi, im);
    }

    // Each control belongs to the following read. Last controls in every
    // sequence receive zero, including singleton sequences.
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "ptx-kernel" fn diagonal_back(
        dmix: *mut f32,
        dgain: *mut f32,
        dbias: *mut f32,
        dread: *const f32,
        mixed: *const f32,
        gain: *const f32,
        width: u32,
        length: u32,
    ) {
        let (r, j, stride, lo, hi, t, at, other): (u32, u32, u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        mov.u32(j, tid.x);
        shl.b32(stride, width, 1);
        mad.lo.u32(lo, r, stride, j);
        add.u32(hi, lo, width);
        rem.u32(t, r, length);
        let (u, v, x, y, a, b, dr, di): (f32, f32, f32, f32, f32, f32, f32, f32);
        load(u, dread, lo);
        load(v, dread, hi);
        let prior: pred;
        setp.ne.u32(prior, t, 0);
        if prior {
            sub.u32(at, lo, stride);
            sub.u32(other, hi, stride);
            load(a, gain, at);
            load(b, gain, other);
            add.rn.f32(a, a, 1.0);
            mul.rn.f32(dr, b, v);
            fma.rn.f32(dr, a, u, dr);
            mul.rn.f32(di, b, u);
            neg.f32(di, di);
            fma.rn.f32(di, a, v, di);
        } else {
            mov.f32(dr, u);
            mov.f32(di, v);
        }
        store(dmix, lo, dr);
        store(dmix, hi, di);
        add.u32(t, t, 1);
        let later: pred;
        setp.lt.u32(later, t, length);
        if later {
            add.u32(at, lo, stride);
            add.u32(other, hi, stride);
            load(u, dread, at);
            load(v, dread, other);
            load(x, mixed, at);
            load(y, mixed, other);
            mul.rn.f32(dr, v, y);
            fma.rn.f32(dr, u, x, dr);
            mul.rn.f32(di, u, y);
            neg.f32(di, di);
            fma.rn.f32(di, v, x, di);
        } else {
            mov.f32(u, 0.0);
            mov.f32(v, 0.0);
            mov.f32(dr, 0.0);
            mov.f32(di, 0.0);
        }
        store(dgain, lo, dr);
        store(dgain, hi, di);
        store(dbias, lo, u);
        store(dbias, hi, v);
    }

    pub unsafe extern "ptx-kernel" fn value_back(
        dvalue: *mut f32,
        dmix: *const f32,
        kernel: *const f32,
        width: u32,
        length: u32,
    ) {
        let (r, k, stride, lo, hi, t, at, other, j, off, bank): (
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
        mov.u32(r, ctaid.x);
        mov.u32(k, tid.x);
        shl.b32(stride, width, 1);
        mad.lo.u32(lo, r, stride, k);
        add.u32(hi, lo, width);
        let (u, v, x, y, w): (f32, f32, f32, f32, f32);
        load(u, dmix, lo);
        load(v, dmix, hi);
        rem.u32(t, r, length);
        let first: pred;
        setp.eq.u32(first, t, 0);
        if first {
            bra(write);
        }
        sub.u32(bank, stride, 1);
        sub.u32(t, r, 1);
        mad.lo.u32(off, t, bank, k);
        add.u32(off, off, width);
        sub.u32(off, off, 1);
        mul.lo.u32(at, r, stride);
        add.u32(other, at, width);
        mov.u32(j, 0);
        label(transpose);
        load(w, kernel, off);
        load(x, dmix, at);
        load(y, dmix, other);
        fma.rn.f32(u, w, x, u);
        fma.rn.f32(v, w, y, v);
        sub.u32(off, off, 1);
        add.u32(at, at, 1);
        add.u32(other, other, 1);
        add.u32(j, j, 1);
        let more: pred;
        setp.lt.u32(more, j, width);
        if more {
            bra(transpose);
        }
        label(write);
        store(dvalue, lo, u);
        store(dvalue, hi, v);
    }

    // Only valid source and destination channel pairs contribute to a diagonal.
    pub unsafe extern "ptx-kernel" fn stencil_back(
        dkernel: *mut f32,
        dmix: *const f32,
        value: *const f32,
        width: u32,
        length: u32,
    ) {
        let (r, z, stride, center, bank, t, at, j, k, xat, yat, tmp): (
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
        mov.u32(r, ctaid.x);
        mov.u32(z, tid.x);
        shl.b32(stride, width, 1);
        sub.u32(center, width, 1);
        sub.u32(bank, stride, 1);
        mad.lo.u32(at, r, bank, z);
        rem.u32(t, r, length);
        add.u32(t, t, 1);
        let last: pred;
        setp.eq.u32(last, t, length);
        let (sum, x, y): (f32, f32, f32);
        mov.f32(sum, 0.0);
        if last {
            bra(write);
        }
        min.u32(tmp, z, center);
        sub.u32(j, center, tmp);
        sub.u32(k, z, tmp);
        add.u32(t, r, 1);
        mul.lo.u32(t, t, stride);
        label(diagonal);
        add.u32(xat, t, k);
        add.u32(yat, t, j);
        load(x, value, xat);
        load(y, dmix, yat);
        fma.rn.f32(sum, x, y, sum);
        add.u32(xat, xat, width);
        add.u32(yat, yat, width);
        load(x, value, xat);
        load(y, dmix, yat);
        fma.rn.f32(sum, x, y, sum);
        add.u32(j, j, 1);
        add.u32(k, k, 1);
        max.u32(tmp, j, k);
        let more: pred;
        setp.lt.u32(more, tmp, width);
        if more {
            bra(diagonal);
        }
        label(write);
        store(dkernel, at, sum);
    }
}

//! Causal vocabulary attention. One warp owns an output row. The transpose
//! gathers at most two contributions per source row without cross-example reads.
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
    fn sum_warp(x: f32) {
        let other: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(other, x, 1 << j, 31, 0xffffffff);
            add.rn.f32(x, x, other);
        }
    }
    fn pair(row: u32, prev: u32, off: u32, tokens: u64, p: u32, length: u32) {
        let (pos, a, b): (u32, u32, u32);
        mov.u32(row, ctaid.x);
        rem.u32(pos, row, length);
        let first: pred;
        setp.eq.u32(first, pos, 0);
        sub.u32(prev, row, 1);
        selp.u32(prev, row, prev, first);
        let (addr, product, modulus): (u64, u64, u64);
        mad.wide.u32(addr, prev, 4, tokens);
        ld.global.u32(a, [addr]);
        mad.wide.u32(addr, row, 4, tokens);
        ld.global.u32(b, [addr]);
        mul.wide.u32(product, a, b);
        cvt.u64.u32(modulus, p);
        rem.u64(product, product, modulus);
        cvt.u32.u64(off, product);
    }
    fn shifted(at: u32, row: u32, s: u32, off: u32, p: u32) {
        add.u32(at, s, off);
        rem.u32(at, at, p);
        mad.lo.u32(at, row, p, at);
    }
    pub unsafe extern "ptx-kernel" fn weights(
        out: *mut f32,
        scores: *const f32,
        tokens: *const u32,
        p: u32,
        length: u32,
    ) {
        let (row, prev, off, s, at): (u32, u32, u32, u32, u32);
        pair(row, prev, off, tokens, p, length);
        let (top, x, z): (f32, f32, f32);
        let done: pred;
        mov.f32(top, f32::NEG_INFINITY);
        mov.u32(s, tid.x);
        label(maxima);
        setp.ge.u32(done, s, p);
        if done {
            bra(reduce_max);
        }
        mad.lo.u32(at, prev, p, s);
        load(x, scores, at);
        max.f32(top, top, x);
        add.u32(s, s, 32);
        bra(maxima);
        label(reduce_max);
        for j in 0..5 {
            shfl.sync.bfly.b32(x, top, 1 << j, 31, 0xffffffff);
            max.f32(top, top, x);
        }
        mov.f32(z, 0.0);
        mov.u32(s, tid.x);
        label(exponents);
        setp.ge.u32(done, s, p);
        if done {
            bra(reduce_mass);
        }
        mad.lo.u32(at, prev, p, s);
        load(x, scores, at);
        sub.rn.f32(x, x, top);
        mul.rn.f32(x, x, 1.4426950408889634);
        ex2.approx.f32(x, x);
        add.rn.f32(z, z, x);
        shifted(at, row, s, off, p);
        store(out, at, x);
        add.u32(s, s, 32);
        bra(exponents);
        label(reduce_mass);
        sum_warp(z);
        mov.u32(s, tid.x);
        label(normalize);
        setp.ge.u32(done, s, p);
        if done {
            ret();
        }
        shifted(at, row, s, off, p);
        load(x, out, at);
        div.rn.f32(x, x, z);
        store(out, at, x);
        add.u32(s, s, 32);
        bra(normalize);
    }
    // Move two coefficients of the tied embedding read. Leave the softmax
    // masses in their own buffer for the derivative with respect to scores.
    pub unsafe extern "ptx-kernel" fn displace(
        effective: *mut f32,
        masses: *const f32,
        tokens: *const u32,
        keys: *const u32,
        p: u32,
        tables: u32,
        strength: f32,
    ) {
        let (row, col, stride, at, b, r): (u32, u32, u32, u32, u32, u32);
        let addr: u64;
        mov.u32(row, ctaid.x);
        mad.wide.u32(addr, row, 4, tokens);
        ld.global.u32(b, [addr]);
        mul.lo.u32(at, row, tables);
        mad.wide.u32(addr, at, 4, keys);
        ld.global.u32(r, [addr]);
        mov.u32(col, tid.x);
        mov.u32(stride, ntid.x);
        let (done, same, add_value, sub_key): (pred, pred, pred, pred);
        setp.eq.u32(same, b, r);
        let x: f32;
        label(column);
        setp.ge.u32(done, col, p);
        if done {
            ret();
        }
        mad.lo.u32(at, row, p, col);
        load(x, masses, at);
        if same {
            bra(write);
        }
        setp.eq.u32(add_value, col, r);
        if add_value {
            add.rn.f32(x, x, strength);
        }
        setp.eq.u32(sub_key, col, b);
        if sub_key {
            sub.rn.f32(x, x, strength);
        }
        label(write);
        store(effective, at, x);
        add.u32(col, col, stride);
        bra(column);
    }
    pub unsafe extern "ptx-kernel" fn pullback(
        unshifted: *mut f32,
        masses: *const f32,
        dw: *const f32,
        tokens: *const u32,
        p: u32,
        length: u32,
    ) {
        let (row, prev, off, s, at): (u32, u32, u32, u32, u32);
        pair(row, prev, off, tokens, p, length);
        let (w, x, mean): (f32, f32, f32);
        let done: pred;
        mov.f32(mean, 0.0);
        mov.u32(s, tid.x);
        label(expectation);
        setp.ge.u32(done, s, p);
        if done {
            bra(reduce);
        }
        mad.lo.u32(at, row, p, s);
        load(w, masses, at);
        load(x, dw, at);
        fma.rn.f32(mean, w, x, mean);
        add.u32(s, s, 32);
        bra(expectation);
        label(reduce);
        sum_warp(mean);
        mov.u32(s, tid.x);
        label(scatter);
        setp.ge.u32(done, s, p);
        if done {
            ret();
        }
        shifted(at, row, s, off, p);
        load(w, masses, at);
        load(x, dw, at);
        sub.rn.f32(x, x, mean);
        mul.rn.f32(x, w, x);
        mad.lo.u32(at, row, p, s);
        store(unshifted, at, x);
        add.u32(s, s, 32);
        bra(scatter);
    }
    pub unsafe extern "ptx-kernel" fn add_sources(
        dscores: *mut f32,
        derivatives: *const f32,
        p: u32,
        length: u32,
    ) {
        let (row, pos, s, at, after, stride): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        rem.u32(pos, row, length);
        let (first, next, done): (pred, pred, pred);
        setp.eq.u32(first, pos, 0);
        add.u32(after, pos, 1);
        setp.lt.u32(next, after, length);
        mov.u32(s, tid.x);
        mov.u32(stride, ntid.x);
        let (x, v): (f32, f32);
        label(col);
        setp.ge.u32(done, s, p);
        if done {
            ret();
        }
        mad.lo.u32(at, row, p, s);
        load(x, dscores, at);
        if first {
            load(v, derivatives, at);
            add.rn.f32(x, x, v);
        }
        if next {
            add.u32(after, at, p);
            load(v, derivatives, after);
            add.rn.f32(x, x, v);
        }
        store(dscores, at, x);
        add.u32(s, s, stride);
        bra(col);
    }
    fn retrieval_factor(factor: f32, row: u32, tokens: u64, keys: u64, tables: u32) {
        let (b, r, at): (u32, u32, u32);
        let addr: u64;
        mad.wide.u32(addr, row, 4, tokens);
        ld.global.u32(b, [addr]);
        mul.lo.u32(at, row, tables);
        mad.wide.u32(addr, at, 4, keys);
        ld.global.u32(r, [addr]);
        let same: pred;
        setp.eq.u32(same, r, b);
        selp.f32(factor, 1.0, 2.0, same);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn combine<const HIT_GAIN: bool>(
        out: *mut f32,
        scaled: *mut f32,
        context: *const f32,
        value: *const f32,
        gain: *const f32,
        tokens: *const u32,
        keys: *const u32,
        tables: u32,
        width: u32,
    ) {
        let (row, col, at): (u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(col, tid.x);
        mad.lo.u32(at, row, width, col);
        let (v, g, x): (f32, f32, f32);
        ld.global.f32(g, [gain]);
        if HIT_GAIN {
            let factor: f32;
            retrieval_factor(factor, row, tokens, keys, tables);
            mul.rn.f32(g, g, factor);
        }
        load(v, value, at);
        mul.rn.f32(v, v, g);
        store(scaled, at, v);
        load(x, context, at);
        add.rn.f32(x, x, v);
        store(out, at, x);
    }
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn gain_back<const HIT_GAIN: bool>(
        partial: *mut f32,
        gradient: *mut f32,
        value: *const f32,
        gain: *const f32,
        tokens: *const u32,
        keys: *const u32,
        tables: u32,
        width: u32,
    ) {
        let (row, col, at): (u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(col, tid.x);
        let (v, g, x, sum): (f32, f32, f32, f32);
        ld.global.f32(g, [gain]);
        let factor: f32;
        if HIT_GAIN {
            retrieval_factor(factor, row, tokens, keys, tables);
            mul.rn.f32(g, g, factor);
        }
        mov.f32(sum, 0.0);
        let done: pred;
        label(col_loop);
        setp.ge.u32(done, col, width);
        if done {
            bra(reduce);
        }
        mad.lo.u32(at, row, width, col);
        load(x, gradient, at);
        load(v, value, at);
        fma.rn.f32(sum, x, v, sum);
        mul.rn.f32(x, x, g);
        store(gradient, at, x);
        add.u32(col, col, 32);
        bra(col_loop);
        label(reduce);
        sum_warp(sum);
        if HIT_GAIN {
            mul.rn.f32(sum, sum, factor);
        }
        mov.u32(col, tid.x);
        setp.ne.u32(done, col, 0);
        if done {
            ret();
        }
        store(partial, row, sum);
    }
    pub unsafe extern "ptx-kernel" fn gain_sum(out: *mut f32, partial: *const f32, rows: u32) {
        let i: u32;
        mov.u32(i, tid.x);
        let (sum, v): (f32, f32);
        mov.f32(sum, 0.0);
        let done: pred;
        label(row_loop);
        setp.ge.u32(done, i, rows);
        if done {
            bra(reduce);
        }
        load(v, partial, i);
        add.rn.f32(sum, sum, v);
        add.u32(i, i, 32);
        bra(row_loop);
        label(reduce);
        sum_warp(sum);
        mov.u32(i, tid.x);
        setp.ne.u32(done, i, 0);
        if done {
            ret();
        }
        st.global.f32([out], sum);
    }
}

//! One warp per 32-dimensional read. The final eligible key is at t - 2.
#[kernite_macros::kernels]
pub mod kernels {
    fn load(v: f32, p: u64, index: u32) {
        let addr: u64;
        mad.wide.u32(addr, index, 4, p);
        ld.global.f32(v, [addr]);
    }
    fn store(p: u64, index: u32, v: f32) {
        let addr: u64;
        mad.wide.u32(addr, index, 4, p);
        st.global.f32([addr], v);
    }
    fn dot(v: f32, x: f32, y: f32) {
        mul.rn.f32(v, x, y);
        let other: f32;
        for j in 0..5 {
            shfl.sync.bfly.b32(other, v, 1 << j, 31, 0xffffffff);
            add.rn.f32(v, v, other);
        }
        mul.rn.f32(v, v, 0.1767766952966369); // 1 / sqrt(32)
    }
    fn exp(v: f32) {
        mul.rn.f32(v, v, 1.4426950408889634);
        ex2.approx.f32(v, v);
    }
    // qkv is [batch, length, 3, 32]. key points at K(h_0) for this lane
    // and sequence. V(h_{i+1}) is 96 + 32 coordinates beyond K(h_i).
    fn indices(q: u32, key: u32, weights: u32, count: u32, length: u32) {
        let (row, t, base, lane): (u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(t, row, length);
        sub.u32(base, row, t);
        mad.lo.u32(q, row, 96, lane);
        mad.lo.u32(key, base, 96, lane);
        add.u32(key, key, 32);
        mul.lo.u32(weights, row, length);
        max.u32(count, t, 1);
        sub.u32(count, count, 1);
    }

    pub unsafe extern "ptx-kernel" fn forward(
        read: *mut f32,
        prob: *mut f32,
        scores: *mut f32,
        qkv: *const f32,
        length: u32,
    ) {
        let (q_at, key, weights, count, i, at, row, lane): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        indices(q_at, key, weights, count, length);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        let (done, leader): (pred, pred);
        setp.eq.u32(leader, lane, 0);
        let (q, k, score, maximum, peak, total, old, mass, value, result): (
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
        load(q, qkv, q_at);
        mov.f32(maximum, f32::NEG_INFINITY);
        mov.f32(total, 0.0);
        mov.u32(i, 0);
        label(score_loop);
        setp.ge.u32(done, i, count);
        if done {
            bra(mix);
        }
        mad.lo.u32(at, i, 96, key);
        load(k, qkv, at);
        dot(score, q, k);
        if leader {
            add.u32(at, weights, i);
            store(scores, at, score);
        }
        max.f32(peak, maximum, score);
        sub.rn.f32(old, maximum, peak);
        exp(old);
        sub.rn.f32(mass, score, peak);
        exp(mass);
        fma.rn.f32(total, total, old, mass);
        mov.f32(maximum, peak);
        add.u32(i, i, 1);
        bra(score_loop);

        label(mix);
        bar.warp.sync(0xffffffff);
        mov.f32(result, 0.0);
        mov.u32(i, 0);
        label(value_loop);
        setp.ge.u32(done, i, count);
        if done {
            bra(finish);
        }
        add.u32(at, weights, i);
        load(mass, scores, at);
        sub.rn.f32(mass, mass, maximum);
        exp(mass);
        div.rn.f32(mass, mass, total);
        if leader {
            store(prob, at, mass);
        }
        mad.lo.u32(at, i, 96, key);
        add.u32(at, at, 128);
        load(value, qkv, at);
        fma.rn.f32(result, mass, value, result);
        add.u32(i, i, 1);
        bra(value_loop);
        label(finish);
        mad.lo.u32(at, row, 32, lane);
        store(read, at, result);
    }

    // dscore_i = p_i <dread, V_{i+1} - read>/sqrt(32), including the
    // dot-product scale. The next kernel gathers each source's derivatives.
    pub unsafe extern "ptx-kernel" fn queries_backward(
        dqkv: *mut f32,
        dscores: *mut f32,
        qkv: *const f32,
        prob: *const f32,
        read: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (q_at, key, weights, count, i, at, row, lane): (
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
        );
        indices(q_at, key, weights, count, length);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        mad.lo.u32(at, row, 32, lane);
        let (grad, output, value, k, p, ds, dq): (f32, f32, f32, f32, f32, f32, f32);
        load(grad, dread, at);
        load(output, read, at);
        mov.f32(dq, 0.0);
        mov.u32(i, 0);
        let (done, leader): (pred, pred);
        setp.eq.u32(leader, lane, 0);
        label(scan);
        setp.ge.u32(done, i, count);
        if done {
            bra(finish);
        }
        mad.lo.u32(at, i, 96, key);
        load(k, qkv, at);
        add.u32(at, at, 128);
        load(value, qkv, at);
        sub.rn.f32(value, value, output);
        dot(ds, grad, value);
        add.u32(at, weights, i);
        load(p, prob, at);
        mul.rn.f32(ds, ds, p);
        if leader {
            store(dscores, at, ds);
        }
        fma.rn.f32(dq, ds, k, dq);
        add.u32(i, i, 1);
        bra(scan);
        label(finish);
        store(dqkv, q_at, dq);
    }

    pub unsafe extern "ptx-kernel" fn sources_backward(
        dqkv: *mut f32,
        qkv: *const f32,
        prob: *const f32,
        dscores: *const f32,
        dread: *const f32,
        length: u32,
    ) {
        let (row, lane, pos, query, end, at): (u32, u32, u32, u32, u32, u32);
        mov.u32(row, ctaid.x);
        mov.u32(lane, tid.x);
        rem.u32(pos, row, length);
        sub.u32(end, row, pos);
        add.u32(end, end, length);
        let (p, g, dk, dv): (f32, f32, f32, f32);
        mov.f32(dk, 0.0);
        mov.f32(dv, 0.0);
        let done: pred;
        add.u32(query, row, 2);
        label(keys);
        setp.ge.u32(done, query, end);
        if done {
            bra(values);
        }
        mad.lo.u32(at, query, length, pos);
        load(p, dscores, at);
        mad.lo.u32(at, query, 96, lane);
        load(g, qkv, at);
        fma.rn.f32(dk, p, g, dk);
        add.u32(query, query, 1);
        bra(keys);
        label(values);
        setp.eq.u32(done, pos, 0);
        if done {
            bra(finish);
        }
        add.u32(query, row, 1);
        label(next_value);
        setp.ge.u32(done, query, end);
        if done {
            bra(finish);
        }
        mad.lo.u32(at, query, length, pos);
        sub.u32(at, at, 1);
        load(p, prob, at);
        mad.lo.u32(at, query, 32, lane);
        load(g, dread, at);
        fma.rn.f32(dv, p, g, dv);
        add.u32(query, query, 1);
        bra(next_value);
        label(finish);
        mad.lo.u32(at, row, 96, lane);
        add.u32(at, at, 32);
        store(dqkv, at, dk);
        add.u32(at, at, 32);
        store(dqkv, at, dv);
    }
}

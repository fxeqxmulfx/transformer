//! Prefix-conditioned cyclic transport of learned complex coordinates.
use crate::{Config, architecture, random::Random};

pub const fn channels(width: usize) -> usize {
    if width < 96 { width } else { 96 }
}

pub const fn parameters(width: usize) -> usize {
    let m = channels(width);
    4 * width + (4 * m + 1) * width + 2 * m + 1
}

pub fn initialize(cfg: &Config, seed: u32, w: &mut [f32]) {
    let d = cfg.width;
    let m = channels(d);
    let start = architecture::memory_start(cfg);
    let control = start + 2 * m * d;
    let bias = control + (2 * m + 1) * d;
    let mut rng = Random::new(seed.wrapping_add(30_000));
    for (range, fan) in [(start..control, 2 * d), (control..bias, d)] {
        for v in &mut w[range] {
            *v = rng.gauss(0., (fan as f64).sqrt().recip()) as f32;
        }
    }
    w[bias..bias + 2 * m + 1].fill(0.);
}

#[kernite_macros::kernels]
pub mod kernels {
    fn load(x: f32, p: u64, i: u32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        ld.global.f32(x, [addr]);
    }
    fn store(p: u64, i: u32, x: f32) {
        let addr: u64;
        mad.wide.u32(addr, i, 4, p);
        st.global.f32([addr], x);
    }
    // One warp normalizes the route distribution and constructs the rotations.
    pub unsafe extern "ptx-kernel" fn prepare(
        route: *mut f32,
        phase: *mut f32,
        control: *const f32,
        bias: *const f32,
        m: u32,
        length: u32,
    ) {
        let (r, lane, t, n, cols, prev, s, at, base, j, k): (
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
        mov.u32(lane, tid.x);
        rem.u32(t, r, length);
        add.u32(n, m, 1);
        add.u32(cols, n, m);
        sub.u32(prev, r, 1);
        mul.lo.u32(prev, prev, cols);
        mul.lo.u32(base, r, n);
        let (v, b, peak, peer, sum, raw, c, si): (f32, f32, f32, f32, f32, f32, f32, f32);
        let (first, done, identity): (pred, pred, pred);
        setp.eq.u32(first, t, 0);
        mov.u32(s, lane);
        if first {
            bra(empty_route);
        }
        mov.f32(peak, f32::NEG_INFINITY);
        label(maximum);
        setp.ge.u32(done, s, n);
        if done {
            bra(reduce_max);
        }
        add.u32(at, prev, s);
        load(v, control, at);
        load(b, bias, s);
        add.rn.f32(v, v, b);
        max.f32(peak, peak, v);
        add.u32(s, s, 32);
        bra(maximum);
        label(reduce_max);
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, peak, 1 << j, 31, 0xffffffff);
            max.f32(peak, peak, peer);
        }
        mov.f32(sum, 0.0);
        mov.u32(s, lane);
        label(exponentials);
        setp.ge.u32(done, s, n);
        if done {
            bra(reduce_sum);
        }
        add.u32(at, prev, s);
        load(v, control, at);
        load(b, bias, s);
        add.rn.f32(v, v, b);
        sub.rn.f32(v, v, peak);
        mul.rn.f32(v, v, 1.4426950408889634f32);
        ex2.approx.f32(raw, v);
        add.u32(at, base, s);
        store(route, at, raw);
        add.rn.f32(sum, sum, raw);
        add.u32(s, s, 32);
        bra(exponentials);
        label(reduce_sum);
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, sum, 1 << j, 31, 0xffffffff);
            add.rn.f32(sum, sum, peer);
        }
        mov.u32(s, lane);
        label(normalize);
        setp.ge.u32(done, s, n);
        if done {
            bra(phases);
        }
        add.u32(at, base, s);
        load(v, route, at);
        div.rn.f32(v, v, sum);
        store(route, at, v);
        add.u32(s, s, 32);
        bra(normalize);
        label(empty_route);
        setp.ge.u32(done, s, n);
        if done {
            bra(phases);
        }
        setp.eq.u32(identity, s, 0);
        selp.f32(v, 1.0, 0.0, identity);
        add.u32(at, base, s);
        store(route, at, v);
        add.u32(s, s, 32);
        bra(empty_route);
        label(phases);
        mov.u32(j, lane);
        mul.lo.u32(base, r, m);
        shl.b32(base, base, 1);
        label(phase_loop);
        setp.ge.u32(done, j, m);
        if done {
            bra(end);
        }
        mov.f32(c, 1.0);
        mov.f32(si, 0.0);
        if first {
            bra(write_phase);
        }
        add.u32(k, n, j);
        add.u32(at, prev, k);
        load(v, control, at);
        load(b, bias, k);
        add.rn.f32(v, v, b);
        cos.approx.f32(c, v);
        sin.approx.f32(si, v);
        label(write_phase);
        add.u32(at, base, j);
        store(phase, at, c);
        add.u32(at, at, m);
        store(phase, at, si);
        add.u32(j, j, 32);
        bra(phase_loop);
        label(end);
    }

    // Forward is a mixture of common shifts. The value VJP uses inverse shifts.
    #[instances(false, true)]
    pub unsafe extern "ptx-kernel" fn mix<const BACK: bool>(
        out: *mut f32,
        input: *const f32,
        route: *const f32,
        phase: *const f32,
    ) {
        let (r, j, m, n, base, k, s, rb, at, lo, hi): (
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
        mov.u32(m, ntid.x);
        add.u32(n, m, 1);
        mul.lo.u32(rb, r, n);
        mul.lo.u32(base, r, m);
        shl.b32(base, base, 1);
        let (re, im, w, x, y, c, si, tmp): (f32, f32, f32, f32, f32, f32, f32, f32);
        mov.f32(re, 0.0);
        mov.f32(im, 0.0);
        if !BACK {
            add.u32(at, rb, m);
            load(re, route, at);
        }
        mov.u32(k, j);
        mov.u32(s, 0);
        let (wrap, done): (pred, pred);
        label(accumulate);
        add.u32(at, rb, s);
        load(w, route, at);
        add.u32(at, base, k);
        load(x, input, at);
        add.u32(at, at, m);
        load(y, input, at);
        fma.rn.f32(re, w, x, re);
        fma.rn.f32(im, w, y, im);
        if BACK {
            setp.eq.u32(wrap, k, 0);
            selp.u32(k, m, k, wrap);
            sub.u32(k, k, 1);
        } else {
            add.u32(k, k, 1);
            setp.eq.u32(wrap, k, m);
            selp.u32(k, 0, k, wrap);
        }
        add.u32(s, s, 1);
        setp.ge.u32(done, s, m);
        if done {
            bra(finish);
        }
        bra(accumulate);
        label(finish);
        add.u32(lo, base, j);
        add.u32(hi, lo, m);
        if !BACK {
            load(c, phase, lo);
            load(si, phase, hi);
            mul.rn.f32(tmp, si, im);
            neg.f32(tmp, tmp);
            fma.rn.f32(tmp, c, re, tmp);
            mul.rn.f32(im, c, im);
            fma.rn.f32(im, si, re, im);
            mov.f32(re, tmp);
        }
        store(out, lo, re);
        store(out, hi, im);
    }

    pub unsafe extern "ptx-kernel" fn unrotate(
        mixed_grad: *mut f32,
        angle_grad: *mut f32,
        grad: *const f32,
        value: *const f32,
        phase: *const f32,
    ) {
        let (r, j, m, at, lo, hi): (u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        mov.u32(j, tid.x);
        mov.u32(m, ntid.x);
        mul.lo.u32(at, r, m);
        shl.b32(lo, at, 1);
        add.u32(lo, lo, j);
        add.u32(hi, lo, m);
        add.u32(at, at, j);
        let (g, h, c, s, re, im, v): (f32, f32, f32, f32, f32, f32, f32);
        load(g, grad, lo);
        load(h, grad, hi);
        load(c, phase, lo);
        load(s, phase, hi);
        load(re, value, lo);
        load(im, value, hi);
        mul.rn.f32(v, g, im);
        neg.f32(v, v);
        fma.rn.f32(v, h, re, v);
        store(angle_grad, at, v);
        mul.rn.f32(v, s, h);
        fma.rn.f32(v, c, g, v);
        store(mixed_grad, lo, v);
        mul.rn.f32(v, s, g);
        neg.f32(v, v);
        fma.rn.f32(v, c, h, v);
        store(mixed_grad, hi, v);
    }

    // Each lane pairs the mixture cotangent with one candidate shifted bank.
    pub unsafe extern "ptx-kernel" fn route_back(
        out: *mut f32,
        grad: *const f32,
        input: *const f32,
        m: u32,
    ) {
        let (r, s, n, base, j, k, at): (u32, u32, u32, u32, u32, u32, u32);
        mov.u32(r, ctaid.x);
        mov.u32(s, tid.x);
        add.u32(n, m, 1);
        mul.lo.u32(base, r, m);
        shl.b32(base, base, 1);
        let (g, h, x, y, total): (f32, f32, f32, f32, f32);
        let (constant, wrap, done): (pred, pred, pred);
        setp.eq.u32(constant, s, m);
        mov.f32(total, 0.0);
        mov.u32(j, 0);
        mov.u32(k, s);
        label(sum);
        add.u32(at, base, j);
        load(g, grad, at);
        if constant {
            bra(one);
        }
        add.u32(at, at, m);
        load(h, grad, at);
        add.u32(at, base, k);
        load(x, input, at);
        add.u32(at, at, m);
        load(y, input, at);
        fma.rn.f32(total, g, x, total);
        fma.rn.f32(total, h, y, total);
        add.u32(k, k, 1);
        setp.eq.u32(wrap, k, m);
        selp.u32(k, 0, k, wrap);
        bra(next);
        label(one);
        add.rn.f32(total, total, g);
        label(next);
        add.u32(j, j, 1);
        setp.ge.u32(done, j, m);
        if done {
            bra(finish);
        }
        bra(sum);
        label(finish);
        mad.lo.u32(at, r, n, s);
        store(out, at, total);
    }

    // The softmax VJP at the next position belongs to this strict-prefix row.
    pub unsafe extern "ptx-kernel" fn controls_back(
        out: *mut f32,
        dr: *const f32,
        route: *const f32,
        angles: *const f32,
        m: u32,
        length: u32,
    ) {
        let (r, lane, n, cols, next, pos, s, k, at, base): (
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
        mov.u32(lane, tid.x);
        add.u32(next, r, 1);
        rem.u32(pos, next, length);
        let (last, done): (pred, pred);
        setp.eq.u32(last, pos, 0);
        add.u32(n, m, 1);
        add.u32(cols, n, m);
        mul.lo.u32(base, r, cols);
        mov.u32(s, lane);
        let (total, peer, p, g): (f32, f32, f32, f32);
        mov.f32(total, 0.0);
        if last {
            bra(write);
        }
        label(dot);
        setp.ge.u32(done, s, n);
        if done {
            bra(reduce);
        }
        mad.lo.u32(at, next, n, s);
        load(g, dr, at);
        load(p, route, at);
        fma.rn.f32(total, g, p, total);
        add.u32(s, s, 32);
        bra(dot);
        label(reduce);
        for j in 0..5 {
            shfl.sync.bfly.b32(peer, total, 1 << j, 31, 0xffffffff);
            add.rn.f32(total, total, peer);
        }
        label(write);
        mov.u32(s, lane);
        label(logits);
        setp.ge.u32(done, s, n);
        if done {
            bra(phase_start);
        }
        mov.f32(g, 0.0);
        if last {
            bra(write_logit);
        }
        mad.lo.u32(at, next, n, s);
        load(g, dr, at);
        load(p, route, at);
        sub.rn.f32(g, g, total);
        mul.rn.f32(g, g, p);
        label(write_logit);
        add.u32(at, base, s);
        store(out, at, g);
        add.u32(s, s, 32);
        bra(logits);
        label(phase_start);
        mov.u32(k, lane);
        add.u32(base, base, n);
        label(phase_loop);
        setp.ge.u32(done, k, m);
        if done {
            bra(end);
        }
        mov.f32(g, 0.0);
        if last {
            bra(write_phase);
        }
        mad.lo.u32(at, next, m, k);
        load(g, angles, at);
        label(write_phase);
        add.u32(at, base, k);
        store(out, at, g);
        add.u32(k, k, 32);
        bra(phase_loop);
        label(end);
    }
}

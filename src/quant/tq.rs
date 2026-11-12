//! Gyro trellis types (TQ2_T, TQK4-TQK7, TQP5-TQP7) and the block-Hadamard rotation of the tensors a Gyro file lists.
//! Written from the agention-gyro specification 1.1 (agentionai-group/agention-gyro, Apache License 2.0, AgentionAI).

pub use super::tq_codebook::TQ_CODEBOOK;
use half::f16;

/// State of step `t`: 16 bits of the block's circular stream `qs` (`steps` steps of `k` bits; spec section 3).
#[inline]
pub fn state(qs: &[u8], k: usize, steps: usize, t: usize) -> usize {
    if k == 8 { return (qs[(t + 31) % 32] as usize) << 8 | qs[t] as usize; }
    let (nb, off) = (steps * k / 8, (steps - 1 - t) * k);
    let i0 = off >> 3;
    let w = qs[i0] as usize | (qs[(i0 + 1) % nb] as usize) << 8 | (qs[(i0 + 2) % nb] as usize) << 16;
    (w >> (off & 7)) & 0xFFFF
}

/// Codebook points of a state: pair 0 and pair 1 (section 5).
#[inline]
pub fn index(s: usize) -> (usize, usize) {
    let x = ((s & 0x7FFF) as u32).wrapping_mul(0x9E3779B1);
    ((x >> 21) as usize, ((x >> 10) & 2047) as usize)
}

#[inline]
pub fn cb(i: usize) -> f32 { f16::from_bits(TQ_CODEBOOK[i]).to_f32() }

/// One 128-weight block: `k` bits per step, `v` weights per step (4, or 2 for the pair trellis TQP).
pub fn dequant_block(b: &[u8], y: &mut [f32], k: usize, v: usize) {
    let (d, qs, steps) = (f16::from_le_bytes([b[0], b[1]]).to_f32(), &b[2..], 128 / v);
    for t in 0..steps {
        let (i0, i1) = index(state(qs, k, steps, t));
        y[v * t] = d * cb(2 * i0);
        y[v * t + 1] = d * cb(2 * i0 + 1);
        if v == 4 {
            y[4 * t + 2] = d * cb(2 * i1);
            y[4 * t + 3] = d * cb(2 * i1 + 1);
        }
    }
}

/// v <- H_b v on each slice of `b` values: the normalised Sylvester-Walsh-Hadamard transform (section 8.1).
pub fn hadamard(v: &mut [f32], b: usize) {
    let s = 1.0 / (b as f32).sqrt();
    for c in v.chunks_exact_mut(b) {
        let mut h = 1;
        while h < b {
            for i in (0..b).step_by(2 * h) {
                for j in i..i + h { let (x, y) = (c[j], c[j + h]); c[j] = x + y; c[j + h] = x - y; }
            }
            h *= 2;
        }
        for x in c { *x *= s; }
    }
}

/// The rotation of a tensor's input features: ±1 signs, one per feature, and the Hadamard block size.
#[derive(Clone, Copy)]
pub struct Rot<'a> { pub signs: &'a [f32], pub block: usize }

impl Rot<'_> {
    /// x' = H (s ⊙ x): the input a stored (rotated) weight row takes, so that w_stored · x' = w · x.
    pub fn rotate(&self, x: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = x.iter().zip(self.signs).map(|(a, s)| a * s).collect();
        hadamard(&mut v, self.block);
        v
    }
    /// w = s ⊙ (H w_stored): a stored row back in the plain basis.
    pub fn unrotate(&self, w: &mut [f32]) {
        hadamard(w, self.block);
        for (a, s) in w.iter_mut().zip(self.signs) { *a *= s; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The worked examples of the specification (section 10): TQK4, TQ2_T, TQP5 blocks and the rotation.
    #[test]
    fn spec_examples() {
        let hex = |s: &str| -> Vec<u8> { s.split_whitespace().map(|b| u8::from_str_radix(b, 16).unwrap()).collect() };
        let near = |a: f32, b: f32| (a - b).abs() < 1e-6 * b.abs().max(1.0);
        let mut y = [0f32; 128];
        let b = hex("00 3c 07 18 29 3a 4b 5c 6d 7e 8f a0 b1 c2 d3 e4 f5 06");
        assert_eq!((state(&b[2..], 4, 32, 0), state(&b[2..], 4, 32, 31)), (0x8070, 0x1807));
        assert_eq!(index(0x8070), (450, 335));
        dequant_block(&b, &mut y, 4, 4);
        for (g, w) in y[..8].iter().zip([0.986816, 0.455322, -0.211914, 0.653809, -0.363281, -0.655762, 1.682617, 0.349121]) { assert!(near(*g, w), "{g} {w}"); }
        for (g, w) in y[124..].iter().zip([1.384766, 2.029297, 0.327148, -1.681641]) { assert!(near(*g, w), "{g} {w}"); }
        let b = hex("00 38 0b 30 55 7a 9f c4 e9 0e 33 58 7d a2 c7 ec 11 36 5b 80 a5 ca ef 14 39 5e 83 a8 cd f2 17 3c 61 86");
        assert_eq!(state(&b[2..], 8, 32, 0), 0x860b);
        dequant_block(&b, &mut y, 8, 4);
        for (g, w) in y[..4].iter().zip([0.137817, 1.110352, -0.016190, 0.334961]) { assert!(near(*g, w), "{g} {w}"); }
        for (g, w) in y[124..].iter().zip([-0.215820, -0.317139, 1.118164, -0.260742]) { assert!(near(*g, w), "{g} {w}"); }
        let b = hex("00 c0 03 68 cd 32 97 fc 61 c6 2b 90 f5 5a bf 24 89 ee 53 b8 1d 82 e7 4c b1 16 7b e0 45 aa 0f 74 d9 3e a3 08 6d d2 37 9c 01 66");
        assert_eq!(state(&b[2..], 5, 64, 0), 0x006c);
        dequant_block(&b, &mut y, 5, 2);
        for (g, w) in y[..4].iter().zip([0.792969, -2.251953, 2.339844, 0.539062]) { assert!(near(*g, w), "{g} {w}"); }
        for (g, w) in y[126..].iter().zip([2.078125, -1.284180]) { assert!(near(*g, w), "{g} {w}"); }
        let r = Rot { signs: &[1.0, -1.0, 1.0, 1.0], block: 4 };
        let mut v = r.rotate(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(v, [3.0, 1.0, -4.0, 2.0]);
        r.unrotate(&mut v);
        assert_eq!(v, [1.0, 2.0, 3.0, 4.0]);
    }
}

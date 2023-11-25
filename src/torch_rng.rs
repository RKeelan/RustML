use rand::{Rng, RngCore, Error};

const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UMASK: u32 = 0x80000000;
const LMASK: u32 = 0x7fffffff;

const FLOAT_MASK: u32 = (1 << 24) - 1;
const FLOAT_DIVISOR: f32 = 1.0f32 / (1 << 24) as f32;

// This type is meant to replicate the behaviour of PyTorch's MT19937RNGEngine, so that this library will be "manual
// seed compatible" with PyTorch. That is, using this library and PyTorch with the same seed should produce the same
// results
pub struct TorchRng {
    seed: u64,
    left: i32,
    seeded: bool,
    next: u32,
    state: [u32; MERSENNE_STATE_N as usize],
}

impl TorchRng {
    pub fn new(seed: u64) -> Self {
        let mut rng = Self {
            seed: seed,
            left: 1,
            seeded: true,
            next: 0,
            state: [0; MERSENNE_STATE_N],
        };
        rng.state[0] = (seed & 0xffffffff) as u32;
        for i in 1..MERSENNE_STATE_N {
            rng.state[i] = 1812433253u32.wrapping_mul(rng.state[i - 1] ^ (rng.state[i - 1] >> 30)) + (i as u32);
        }
        rng
    }

    pub fn torch_gen(&mut self) -> f32 {
        ((self.gen::<u32>() & FLOAT_MASK) as f32) * FLOAT_DIVISOR
    }

    fn next_state(&mut self) {
        let mut idx: usize = 0;
        self.left = MERSENNE_STATE_N as i32;
        self.next = 0;

        for _ in (1..(MERSENNE_STATE_N - MERSENNE_STATE_M + 1)).rev() {
            self.state[idx] = self.state[idx + MERSENNE_STATE_M] ^ 
                TorchRng::twist(self.state[idx], self.state[idx + 1]);
            idx += 1;
        }

        for _ in (1..MERSENNE_STATE_M).rev() {
            self.state[idx] = self.state[idx + MERSENNE_STATE_M - MERSENNE_STATE_N] ^
                TorchRng::twist(self.state[idx], self.state[idx + 1]);
            idx += 1;
        }

        self.state[idx] = self.state[idx + MERSENNE_STATE_M - MERSENNE_STATE_N] ^
            TorchRng::twist(self.state[idx], self.state[0]);
    }

    fn twist(u: u32, v: u32) -> u32 {
        (TorchRng::mix_bits(u, v) >> 1) ^ (if (v & 1) != 0 { MATRIX_A } else { 0 })
    }

    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }
}

impl RngCore for TorchRng {
    fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next as usize];
        self.next += 1;
        y ^= y >> 11;
        y ^= y << 7 & 0x9d2c5680;
        y ^= y << 15 & 0xefc60000;
        y ^= y >> 18;
        y
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u32() as u64
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        panic!("TorchRng does not implement fill_bytes");
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), Error> {
        panic!("TorchRng does not implement try_fill_bytes");
    }
}
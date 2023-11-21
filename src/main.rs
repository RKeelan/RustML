use ml::Tensor;
use rand::Rng;
use rand_mt::{Mt,Mt64};

// This binary used for exploratory testing
fn main() {
    let seed = 2147483647;
    let mut rng32 = Mt::new(seed);
    let rand32: Vec<f32> = (0..5).map(|_| rng32.gen_range(0.0..1.0)).collect();
    let r32_strings = rand32.iter().map(|x| format!("{:.4}", x)).collect::<Vec<String>>();
    println!("Rust32:          {}", r32_strings.join(", "));
    
    let mut rng64 = Mt64::new(seed as u64);
    let rand64: Vec<f32> = (0..5).map(|_| rng64.gen_range(0.0..1.0)).collect();
    let r64_strings = rand64.iter().map(|x| format!("{:.4}", x)).collect::<Vec<String>>();
    println!("Rust64:          {}", r64_strings.join(", "));
    // I was able to get Numpy and Python to produce the same set of numbers. The 32-bit implementation of Mt for Rust
    // appears to produce the same number as Python / Numpy for every other number. Rust64 and PyTorch are totally
    // different.
    /*
Numpy:           0.3934, 0.6567, 0.0863, 0.9644, 0.6211
Python:          0.3934, 0.6567, 0.0863, 0.9644, 0.6211
Rust32:          0.3934, 0.8920, 0.6567, 0.6391, 0.0863
Rust64:          0.6839, 0.9944, 0.8243, 0.5297, 0.0530
PyTorch: tensor([0.7081, 0.3542, 0.1054, 0.5996, 0.0904])
    */
}
use ml::torch_rng::TorchRng;

// This binary used for exploratory testing
fn main() {
    let seed = 2147483647;
    let mut rng = TorchRng::new(seed as u64);
    let torch32: Vec<f32> = (0..5).map(|_| rng.torch_gen()).collect();
    let t32_strings = torch32.iter().map(|x| format!("{:.7}", x)).collect::<Vec<String>>();
    println!("Torch32: {}", t32_strings.join(", "));
    /*
Numpy:   0.3933911, 0.6566618, 0.0862908, 0.9643838, 0.6211251
Python:  0.3933911, 0.6566618, 0.0862908, 0.9643838, 0.6211251
PyTorch: 0.7081289, 0.3542391, 0.1054323, 0.5996444, 0.0904441
Torch32: 0.7081289, 0.3542391, 0.1054323, 0.5996444, 0.0904441
    */
}
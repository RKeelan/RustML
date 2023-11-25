use ml::torch_rng::TorchRng;
use rand::Rng;
use rand_mt::Mt;


// This binary used for exploratory testing
fn main() {
    let seed = 2147483647;

    let mut trng32 = TorchRng::new(seed as u64);
    //let torch32: Vec<f32> = (0..5).map(|_| trng32.gen_range(0.0..1.0)).collect();
    let torch32: Vec<f32> = (0..5).map(|_| trng32.torch_gen()).collect();
    let t32_strings = torch32.iter().map(|x| format!("{:.7}", x)).collect::<Vec<String>>();
    println!("Torch32: {}", t32_strings.join(", "));

    let mut rng32 = Mt::new(seed);
    let rand32: Vec<f32> = (0..5).map(|_| rng32.gen_range(0.0..1.0)).collect();
    let r32_strings = rand32.iter().map(|x| format!("{:.7}", x)).collect::<Vec<String>>();
    println!("Rust32:  {}", r32_strings.join(", "));
    /*
Numpy:   0.3933911, 0.6566618, 0.0862908, 0.9643838, 0.6211251
Python:  0.3933911, 0.6566618, 0.0862908, 0.9643838, 0.6211251
PyTorch: 0.7081289, 0.3542391, 0.1054323, 0.5996444, 0.0904441
Torch32: 0.7081289, 0.3542391, 0.1054323, 0.5996444, 0.0904441
Rust32:  0.3933910, 0.8920087, 0.6566617, 0.6390611, 0.0862907
    */
}
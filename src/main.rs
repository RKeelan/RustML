use ml::Tensor;
use ml::tensor::REQUIRES_GRAD;

// This binary used for exploratory testing
fn main() {
    unsafe {REQUIRES_GRAD = true;}
    let numerator = Tensor::new_0d(5.0);
    let denominator = Tensor::new_0d(2.0);
    let result = &numerator / &denominator;
    result.bwd().unwrap();

    println!("Numerator: {}", numerator);
    println!("Denominator: {}", denominator);
    println!("Result: {}", result);
}
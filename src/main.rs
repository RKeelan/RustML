use ml::Tensor;
use ml::TensorError;
use ml::tensor::REQUIRES_GRAD;

// This binary used for exploratory testing
fn main() {
    let a = Tensor::new_0d(1);
    let b = Tensor::new_0d(2);
    let c = &a + &b;
    println!("a\n{}", a);
    println!("b\n{}", b);
    println!("c\n{}", c);
}
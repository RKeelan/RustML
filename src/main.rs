use ml::Tensor;
use ml::tensor::REQUIRES_GRAD;

// This binary used for exploratory testing
fn main() {
    unsafe {REQUIRES_GRAD = true;}
    let a = Tensor::new_2d(vec![vec![1.0,5.0],vec![1.0,1.1]]);
    let sum = a.sum();
    sum.bwd().unwrap();

    println!("a: {}", a);
    println!("sum: {}", sum);
}
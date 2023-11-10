use ml::Tensor;
use ml::TensorError;
use ml::tensor::REQUIRES_GRAD;

// This binary used for exploratory testing
fn main() {
    unsafe {REQUIRES_GRAD = true;}
    let a = Tensor::new_0d(1.0_f32);
    let b = Tensor::new_0d(2.0_f32);
    let mut c = &a + &b;
    c.bwd().unwrap();
    println!("a\n{}", a);
    println!("b\n{}", b);
    println!("c\n{}", c);
    
    c.set_requires_grad(false);
    
    unsafe {REQUIRES_GRAD = false;}
    let d = Tensor::new_1d(vec![1.0_f32, 2.0_f32]);
    let e = Tensor::new_1d(vec![3.0_f32, 4.0_f32]);
    let f = &d + &e;
    assert_eq!(f.bwd(), Err(TensorError::NoGrad));
}
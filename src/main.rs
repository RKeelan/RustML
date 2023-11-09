use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let a = Tensor::new_0d(1.0);
    let b = Tensor::new_0d(2.0);
    let c = &a + &b;
    c.bwd();
    println!("a\n{}", a);
    println!("b\n{}", b);
    println!("c\n{}", c);
}
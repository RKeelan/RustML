use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let d0 = Tensor::new_0d(1);
    let d1 = Tensor::new_1d((1..11).collect());
    let d2 = Tensor::new(vec![3,3],(0..9).map(|x| x as f32).collect());
    let d3 = Tensor::new(vec![3,3,3],(0..27).map(|x| x as f32).collect());

    println!("{}", d0);
    println!("{}", d1);
    println!("{}", d2);
    println!("{}", d3);
}
use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let a = Tensor::new_2d(vec![vec![1,2],vec![3,4]]);
    println!("{:?}", a.element([0,0].to_vec()));
}
use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let d0 = Tensor::new_0d(1);
    let d1 = Tensor::new_1d((1..11).collect());
    let d2 = Tensor::new(vec![3,3],(0..9).map(|x| x as f32).collect());
    let d3 = Tensor::new(vec![3,3,3],(0..27).map(|x| x as f32).collect());
    let d4 = Tensor::new(vec![3,3,3,3],(0..81).map(|x| x as f32).collect());

    println!("d0: {}", d0);
    println!("d0[0]: {}", d0.item());
    println!("");
    println!("d1: {}", d1);
    println!("d1[0]: {}", d1.get(vec![0]));
    println!("");
    println!("d2: {}", d2);
    println!("d2[0]: {}", d2.get(vec![0]));
    println!("d2[1]: {}", d2.get(vec![1]));
    println!("d2[2]: {}", d2.get(vec![2]));
    println!("");
    println!("d3: {}", d3);
    println!("d3[0]: {}", d3.get(vec![0]));
    println!("d3[1]: {}", d3.get(vec![1]));
    println!("d3[2]: {}", d3.get(vec![2]));
    println!("");
    println!("d4: {}", d4);
    println!("d4[0]: {}", d4.get(vec![0]));
    println!("d4[1]: {}", d4.get(vec![1]));
    println!("d4[2]: {}", d4.get(vec![2]));
}
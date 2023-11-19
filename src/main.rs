use ml::Tensor;

// This binary used for exploratory testing
fn main() {
    let a = Tensor::new_2d(vec![vec![1.0, 2.0, 4.0], vec![8.0, 32.0, 64.0]], true);
    //let a = Tensor::new_0d(1.0, true);
    let b = Tensor::new_0d(2.0, true);

    let c = &a + &b;
    c.bwd().unwrap();
    // println!("### a + b = c ###");
    // println!("a: {}", a);
    // println!("b: {}", b);
    // println!("c: {}", c);


    let c = &a + &b;
    a.zero_grad(); b.zero_grad();
    c.bwd().unwrap();
    // println!("### b + a = c ###");
    // println!("a: {}", a);
    // println!("b: {}", b);
    // println!("c: {}", c);

    println!("### a / b = c ###");
    let c = &b / &a;
    a.zero_grad(); b.zero_grad();
    c.bwd().unwrap();
    println!("a: {}", a);
    println!("b: {}", b);
    println!("c: {}", c);
}
import argparse
import sys

import torch


def add(name, a, b):
    c = a + b
    c.backward(torch.ones(c.shape))
    print()
    print(f"        // {name}")
    print(f"        let a = Tensor::new(vec!{list(a.size())}, vec!{a.flatten().tolist()}, true);")
    print(f"        let b = Tensor::new(vec!{list(b.size())}, vec!{b.flatten().tolist()}, true);")
    print(f"        let expected_data = vec!{c.flatten().tolist()};")
    print(f"        let a_grad = Some(vec!{a.grad.flatten().tolist()});")
    print(f"        let b_grad = Some(vec!{b.grad.flatten().tolist()});")
    print(f"        check_add(&a, &b, &expected_data, &a_grad, &b_grad);")
    print(f"        a.zero_grad(); b.zero_grad();")
    print(f"        check_add(&a, &b, &expected_data, &a_grad, &b_grad);")


def generate_add_tests():
    add("0d",
        torch.tensor([5.0], requires_grad=True), 
        torch.tensor([6.0], requires_grad=True))
    
    add("1d",
        torch.tensor([1.0, -1.0, 2.0], requires_grad=True), 
        torch.tensor([2.0, 3.0, -6.0], requires_grad=True))
    
    a = torch.FloatTensor(list(range(5,9))).reshape(2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(1,12,3))).reshape(2,2); b.requires_grad=True
    add("2d", a, b)
    
    a = torch.FloatTensor(list(range(0,8))).reshape(2,2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(4,20,2))).reshape(2,2,2); b.requires_grad=True
    add("3d", a, b)
    
    a = torch.FloatTensor(list(range(0,16))).reshape(2,2,2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(8,40,2))).reshape(2,2,2,2); b.requires_grad=True
    add("4d", a, b)


def generate_get_tests():
    a = torch.tensor(list(range(9))).reshape(3,3)
    print(f"a[0,0]={a[0,1]}")
    print(f"a[1,1]={a[1,1]}")
    print(f"a[2,2]={a[2,2]}")
    
    b = torch.tensor(list(range(27))).reshape(3,3,3)
    print(f"b[0,0,2]={b[0,0,2]}")
    print(f"b[1,1,1]={b[1,1,1]}")
    print(f"b[2,0,0]={b[2,0,0]}")
    
    c = torch.tensor(list(range(81))).reshape(3,3,3,3)
    print(f"b[0,0,2,1]={c[0,0,2,1]}")
    print(f"b[1,1,1,0]={c[1,1,1,0]}")
    print(f"b[2,0,0,2]={c[2,0,0,2]}")


def generate_idx_test():
    d4 = torch.tensor(list(range(120))).reshape(2,3,4,5)
    for n in range(2):
        d3 = d4[n]
        for c in range(3):
            d2 = d3[c]
            for h in range(4):
                d1 = d2[h]
                for w in range(5):
                    d0 = d1[w]
                    print(f"d4[{n},{c},{h},{w}]={d0.item()}")
                    assert d0.item() == d4[n,c,h,w].item()


def divide(name, a, b):
    c = a / b
    c.backward(torch.ones(c.shape))
    print()
    print(f"        // {name}")
    print(f"        let a = Tensor::new(vec!{list(a.size())}, vec!{a.flatten().tolist()}, true);")
    print(f"        let b = Tensor::new(vec!{list(b.size())}, vec!{b.flatten().tolist()}, true);")
    print(f"        let expected_data = vec!{c.flatten().tolist()};")
    print(f"        let a_grad = Some(vec!{a.grad.flatten().tolist()});")
    print(f"        let b_grad = Some(vec!{b.grad.flatten().tolist()});")
    print(f"        check_div(&a, &b, &expected_data, &a_grad, &b_grad);")
    print()

    a.grad = torch.zeros(a.shape)
    b.grad = torch.zeros(b.shape)
    c = b / a
    c.backward(torch.ones(c.shape))
    print(f"        a.zero_grad(); b.zero_grad();")
    print(f"        let expected_data = vec!{c.flatten().tolist()};")
    print(f"        let a_grad = Some(vec!{a.grad.flatten().tolist()});")
    print(f"        let b_grad = Some(vec!{b.grad.flatten().tolist()});")
    print(f"        check_div(&b, &a, &expected_data, &b_grad, &a_grad);")


def generate_div_tests():
    divide("0d",
        torch.tensor([5.0], requires_grad=True, dtype=torch.float64), 
        torch.tensor([6.0], requires_grad=True, dtype=torch.float64))
    
    divide("1d",
        torch.tensor([1.0, -1.0, 2.0], requires_grad=True, dtype=torch.float64), 
        torch.tensor([2.0, 3.0, -6.0], requires_grad=True, dtype=torch.float64))
    
    a = torch.FloatTensor(list(range(5,9))).type(torch.DoubleTensor).reshape(2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(1,12,3))).type(torch.DoubleTensor).reshape(2,2); b.requires_grad=True
    divide("2d", a, b)
    
    a = torch.FloatTensor(list(range(0,8))).type(torch.DoubleTensor).reshape(2,2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(4,20,2))).type(torch.DoubleTensor).reshape(2,2,2); b.requires_grad=True
    divide("3d", a, b)
    
    a = torch.FloatTensor(list(range(0,16))).type(torch.DoubleTensor).reshape(2,2,2,2); a.requires_grad=True
    b = torch.FloatTensor(list(range(8,40,2))).type(torch.DoubleTensor).reshape(2,2,2,2); b.requires_grad=True
    divide("4d", a, b)


def generate_broadcast_tests():
    single_0d = torch.tensor(0)
    single_1d = torch.tensor([1])
    single_2d = torch.tensor([[2]])
    single_3d = torch.tensor([[[3]]])
    single_4d = torch.tensor([[[[4]]]])

    multi_1d = torch.tensor([1, 2, 3])
    multi_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
    multi_3d = torch.tensor([[[1, 2, 3],
                             [4, 5, 6]],
                            [[7, 8, 9],
                             [10, 11, 12]]])
    multi_4d = torch.tensor([[[[1, 2, 3],
                              [4, 5, 6]],
                             [[7, 8, 9],
                              [0, 1, 2]]],

                            [[[1, 2, 3],
                              [4, 5, 6]],
                             [[7, 8, 9],
                              [0, 1, 2]]]])
   
    print(f"    let multi_1d = Tensor::new(vec!{list(multi_1d.size())}, vec!{multi_1d.flatten().tolist()}, false);")
    print(f"    let multi_2d = Tensor::new(vec!{list(multi_2d.size())}, vec!{multi_2d.flatten().tolist()}, false);")
    print(f"    let multi_3d = Tensor::new(vec!{list(multi_3d.size())}, vec!{multi_3d.flatten().tolist()}, false);")
    print(f"    let multi_4d = Tensor::new(vec!{list(multi_4d.size())}, vec!{multi_4d.flatten().tolist()}, false);")
    print()
    print(f"    assert_eq!(vec!{list((single_0d + multi_1d).size())}, single_0d.get_broadcasted_shape(&multi_1d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_0d + multi_2d).size())}, single_0d.get_broadcasted_shape(&multi_2d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_0d + multi_3d).size())}, single_0d.get_broadcasted_shape(&multi_3d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_0d + multi_4d).size())}, single_0d.get_broadcasted_shape(&multi_4d).unwrap());")
    print()
    print(f"    assert_eq!(vec!{list((single_1d + multi_1d).size())}, single_1d.get_broadcasted_shape(&multi_1d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_1d + multi_2d).size())}, single_1d.get_broadcasted_shape(&multi_2d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_1d + multi_3d).size())}, single_1d.get_broadcasted_shape(&multi_3d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_1d + multi_4d).size())}, single_1d.get_broadcasted_shape(&multi_4d).unwrap());")
    print()
    print(f"    assert_eq!(vec!{list((single_2d + multi_1d).size())}, single_2d.get_broadcasted_shape(&multi_1d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_2d + multi_2d).size())}, single_2d.get_broadcasted_shape(&multi_2d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_2d + multi_3d).size())}, single_2d.get_broadcasted_shape(&multi_3d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_2d + multi_4d).size())}, single_2d.get_broadcasted_shape(&multi_4d).unwrap());")
    print()
    print(f"    assert_eq!(vec!{list((single_3d + multi_1d).size())}, single_3d.get_broadcasted_shape(&multi_1d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_3d + multi_2d).size())}, single_3d.get_broadcasted_shape(&multi_2d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_3d + multi_3d).size())}, single_3d.get_broadcasted_shape(&multi_3d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_3d + multi_4d).size())}, single_3d.get_broadcasted_shape(&multi_4d).unwrap());")
    print()
    print(f"    assert_eq!(vec!{list((single_4d + multi_1d).size())}, single_4d.get_broadcasted_shape(&multi_1d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_4d + multi_2d).size())}, single_4d.get_broadcasted_shape(&multi_2d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_4d + multi_3d).size())}, single_4d.get_broadcasted_shape(&multi_3d).unwrap());")
    print(f"    assert_eq!(vec!{list((single_4d + multi_4d).size())}, single_4d.get_broadcasted_shape(&multi_4d).unwrap());")
    

def generate_broadcast_op_tests():
    single_0d = torch.tensor(2).float(); single_0d.requires_grad=True
    multi_2d = torch.tensor([[1, 2, 4], [8, 32, 64]]).float(); multi_2d.requires_grad=True
    b = torch.tensor(1).float(); b.requires_grad=True

    # add("Add", single_0d, multi_2d)
    divide("Divide", multi_2d, single_0d)
    single_0d.grad = torch.zeros(single_0d.shape)
    multi_2d.grad = torch.zeros(multi_2d.shape)
    divide("Divide", b, single_0d)


def main(args):
    parser = argparse.ArgumentParser(description="Generate test data for the Rust ml package")
    parser.add_argument("test")
    args = parser.parse_args()

    if args.test == "add":
        generate_add_tests()
    elif args.test == "broadcast":
        generate_broadcast_tests()
    elif args.test == "broadcast-op":
        generate_broadcast_op_tests()
    elif args.test == "div":
        generate_div_tests()
    elif args.test == "get":
        generate_get_tests()
    elif args.test == "idx":
        generate_idx_test()
    else:
        print(f"Unknown test: {args.test}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
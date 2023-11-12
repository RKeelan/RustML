import argparse
import sys

import torch

def add(name, a, b):
    c = a + b
    print()
    print(f"        // {name}")
    print(f"        let a = Tensor::new(vec!{list(a.size())}, vec!{a.flatten().tolist()});")
    print(f"        let b = Tensor::new(vec!{list(b.size())}, vec!{b.flatten().tolist()});")
    print(f"        let expected = vec!{c.flatten().tolist()};")
    print(f"        check_add(&a, &b, expected);")
    # For add, the gradients of a and b are the same as the gradients of c

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



def main(args):
    parser = argparse.ArgumentParser(description="Generate test data for the Rust ml package")
    parser.add_argument("test")
    args = parser.parse_args()

    if args.test == "add":
        generate_add_tests()
    elif args.test == "get":
        generate_get_tests()
    elif args.test == "idx":
        generate_idx_test()
    else:
        print(f"Unknown test: {args.test}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
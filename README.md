# RustML

[![CI](https://github.com/RKeelan/RustML/actions/workflows/ci.yml/badge.svg)](https://github.com/RKeelan/RustML/actions/workflows/ci.yml)

A native Rust, not-(yet?)-production grade implementation of PyTorch for following Andrej's Karpathy's "Neural Networks: Zero to Hero" tutorials in Rust.

## Features

- N-dimensional tensors supporting i8, i16, i32, f32, and f64
- Overloaded operators: +, / (more coming soon)
- Automatic differentiation and back propagation for f32 and f64 tensors
- Limited broadcasting support
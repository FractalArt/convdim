# ConvDim

`ConvDim` is a command-line application written in the [Rust](https://www.rust-lang.org/) programming language that computes the output dimension
of a convolution layer in a convolutional neural network, given the dimension of its input, the size of the convolution
filter as well as its stride and padding.

***

## Building & Usage

`ConvDim` requires the Rust compiler to be available. Instructions on how to install it can be found at on the [Rust website](https://www.rust-lang.org/tools/install).
Once the `Rust` toolchain is setup, the application can be compiled as follows:

```sh
> cargo build --release
> ./target/release/convdim --help
```

The usage is straightforward:

```sh
> ./target/release/convdim --input-dim 28 --filter-size 5 --stride 1 --padding 0
24
```

or similarly using the short versions of the flags:

```sh
> ./target/release/convdim -i 28 -f 5 -s 1 -p 0
24
```

## Documentation

To generate and open the documentation of the code in the web browser, run:

```sh
> cargo doc --open --no-deps
```

## Tests

Unit tests can be run by typing

```sh
> cargo t --release
```

## Licensing
`convdim` is made available under the terms of either the MIT License or the Apache License 2.0, at your option.

See the [LICENSE-APACHE](LICENSE-APACHE) and [LICENSE-MIT](LICENSE-MIT) files for license details.

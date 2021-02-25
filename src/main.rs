//! Compute the dimension of the output of a convolutional layer in a convolutional network.
//!
//! The dimension of the output is computed from the dimension of the input entering the layer, the size
//! of the filter associated to the layer, the stride that is used to slide the filter along
//! the input, as well as the padding that can potentially be applied to the input before performing
//! the convolution.
//!
//! It is assumed that everything (dimensions of the input and the filter as well as stride step and padding)
//! is symmetric in the `x` and `y` directions. If this is not the case, the program can be run twice by specifying
//! the different parameters corresponding to the horizontal and vertical directions separately.
use serde::Deserialize;
use structopt::StructOpt;

#[derive(Deserialize, Debug)]
/// ## A (transposed) convolutional layer.
///
/// It is defined by its `filter_size`, the `padding` that is applied to
/// the input before application of the filter, the `stride` with which the
/// filter moves across the input tensor as well as the information on whether
/// the layer is a convolutional or a transposed convolutional layer.
struct Layer {
    filter_size: u16,
    stride: u16,
    padding: u16,
    transposed: bool,
}

#[derive(Deserialize, Debug)]
/// ## A collection of successive layers.
///
/// This is simply a wrapper around a `Vec<Layer>` that can be
/// deserialized using [`serde`](https://docs.rs/crate/serde/1.0.116).
struct Layers {
    layers: Vec<Layer>,
}

#[derive(Debug, StructOpt)]
/// ## Compute the dimension of the output of a (transposed) convolutional layer.
///
/// It is assumed that the input has squared dimension `input-dim`.
/// If this is not the case, the height `h` and width `w` output dimensions
/// can be computed separately by running the program twice and setting
/// `--input-dim h` and `--input-dim w`.
/// The same argument can be made for the filter.
struct Opt {
    #[structopt(
        short = "t",
        long = "toml",
        parse(from_os_str),
        // Everything except the input dimension is specified in the toml file.
        conflicts_with_all(&["transposed", "filter-size", "padding", "stride", "repeat"])
    )]
    /// Path to the toml file from which the successive layers and the input dimension shall be read.
    toml: Option<std::path::PathBuf>,

    #[structopt(short = "i", long = "input-dim")]
    /// The dimension of input.
    in_dim: u16,

    #[structopt(short = "f", long = "filter-size", default_value = "3")]
    /// The filter size.
    filter_size: u16,

    #[structopt(short = "p", long = "padding", default_value = "0")]
    /// The zero-padding that is used for the filter.
    padding: u16,

    #[structopt(short = "s", long = "stride", default_value = "1")]
    /// The stride that is used for the filter.
    stride: u16,

    #[structopt(short = "r", long = "repeat", default_value = "1")]
    /// The number of times that the convolution layer is applied.
    repeat: u16,

    #[structopt(short = "d", long = "transposed")]
    /// Flag that specifies that the layer is a transposed convolutional layer.
    transposed: bool,
}

/// ## Compute the output dimension of a convolutional layer.
///
/// The dimension of the output (o) of the convolutional layer is computed from
/// its input dimension `in_dim` (n), the size of its filter `filter_size` (f) as well
/// as the zero-`padding` applied to the input and the `stride` that is used to slide
/// the filter according to:
///
/// o = (n - f + 2*p) / s + 1
///
/// ## Example
///
/// ```rust
/// assert_eq!(conv_output_dim(28, 5, 0, 1, 1), 24);
/// ```
fn conv_output_dim(in_dim: u16, filter_size: u16, padding: u16, stride: u16, repeat: u16) -> u16 {
    if filter_size > in_dim + 2 * padding {
        panic!(
            "Filter size ({}) is larger than (padded) input ({})!",
            filter_size, in_dim
        );
    }
    match repeat {
        0 => in_dim,
        1 => (in_dim - filter_size + 2 * padding) / stride + 1,
        n => conv_output_dim(
            (in_dim - filter_size + 2 * padding) / stride + 1,
            filter_size,
            padding,
            stride,
            n - 1,
        ),
    }
}

/// ## Compute the output dimension of a transposed convolutional layer.
///
/// The dimension of the output (o) of the transposed convolutional layer is computed from
/// its input dimension `in_dim` (n), the size of its filter `filter_size` (f) as well
/// as the zero-`padding` applied to the input and the `stride` that is used to slide
/// the filter according to:
///
/// o = (n - 1) * s + f - 2*p
///
/// ## Example
///
/// ```rust
/// assert_eq!(transposed_conv_output_dim(32, 2, 0, 2, 1), 64);
/// ```
fn transposed_conv_output_dim(
    in_dim: u16,
    filter_size: u16,
    padding: u16,
    stride: u16,
    repeat: u16,
) -> u16 {
    if in_dim == 0 {
        panic!("Input to transposed convolutional layer needs to be strictly positive.");
    }

    if (in_dim - 1) * stride + filter_size < 2 * padding {
        panic!("Parameters of the transposed convolutional layer lead to a negative output.");
    }
    match repeat {
        1 => (in_dim - 1) * stride + filter_size - 2 * padding,
        n => transposed_conv_output_dim(
            (in_dim - 1) * stride + filter_size - 2 * padding,
            filter_size,
            padding,
            stride,
            n - 1,
        ),
    }
}

/// ## Compute the dimension after a several consecutive (transposed) convolutional layers.
///
/// This corresponds to computing the output after passing an `in_dim`-dimensional input
/// through all the specified `layers`.
fn dim_after_layers(layers: &[Layer], in_dim: u16) -> u16 {
    layers.iter().fold(in_dim, |intermediate_dim, layer| {
        if layer.transposed {
            transposed_conv_output_dim(
                intermediate_dim,
                layer.filter_size,
                layer.padding,
                layer.stride,
                1,
            )
        } else {
            conv_output_dim(
                intermediate_dim,
                layer.filter_size,
                layer.padding,
                layer.stride,
                1,
            )
        }
    })
}

fn main() {
    let opt = Opt::from_args();

    if let Some(path) = opt.toml {
        // Parse the file content
        let toml_content = match std::fs::read_to_string(&path) {
            Ok(file) => file,
            Err(e) => {
                println!("Unable to open input file '{:?}'", path);
                panic!("{}", e);
            }
        };

        // De-serialize the toml content
        let layers: Layers = match toml::from_str(&toml_content) {
            Ok(layers) => layers,
            Err(e) => {
                panic!("Error reading toml input file: {}", e)
            }
        };

        println!("{}", dim_after_layers(&layers.layers, opt.in_dim));
    } else if opt.transposed {
        println!(
            "{}",
            transposed_conv_output_dim(
                opt.in_dim,
                opt.filter_size,
                opt.padding,
                opt.stride,
                opt.repeat
            )
        );
    } else {
        println!(
            "{}",
            conv_output_dim(
                opt.in_dim,
                opt.filter_size,
                opt.padding,
                opt.stride,
                opt.repeat
            )
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_output_dim() {
        assert_eq!(conv_output_dim(28, 5, 0, 1, 1), 24);
        assert_eq!(conv_output_dim(24, 5, 0, 1, 1), 20);
        assert_eq!(conv_output_dim(28, 5, 0, 1, 2), 20);
        assert_eq!(conv_output_dim(4, 4, 1, 1, 1), 3);
        assert_eq!(conv_output_dim(64, 2, 0, 2, 1), 32);
    }

    #[test]
    fn test_transposed_conv_output_dim() {
        assert_eq!(transposed_conv_output_dim(32, 2, 0, 2, 1), 64);
    }

    #[test]
    fn test_conv_transposed_conv_chain() {
        let in_dim = 64;
        let stride = 2;
        let filter_size = 3;
        let padding = 1;

        let conv_out = conv_output_dim(in_dim, filter_size, padding, stride, 1);
        assert_eq!(conv_out, 32);
        let transposed_conv_out =
            transposed_conv_output_dim(conv_out, filter_size, padding, stride, 1);
        assert_eq!(transposed_conv_out, 63);
    }

    #[test]
    fn test_dim_after_layers() {
        // Convolutional auto-encoder
        let layers = vec![
            // encoder
            Layer {
                filter_size: 3,
                stride: 1,
                padding: 1,
                transposed: false,
            },
            Layer {
                filter_size: 2,
                stride: 2,
                padding: 0,
                transposed: false,
            },
            Layer {
                filter_size: 3,
                stride: 1,
                padding: 1,
                transposed: false,
            },
            Layer {
                filter_size: 2,
                stride: 2,
                padding: 0,
                transposed: false,
            },
            // decoder
            Layer {
                filter_size: 2,
                stride: 2,
                padding: 0,
                transposed: true,
            },
            Layer {
                filter_size: 2,
                stride: 2,
                padding: 0,
                transposed: true,
            },
        ];

        assert_eq!(dim_after_layers(&layers, 64), 64);
    }
}

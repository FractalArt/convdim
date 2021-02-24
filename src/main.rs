//! Compute the dimension of the output of a convolutional layer in a convolutional network.
//!
//! The dimension of the output is computed from the dimension of the input entering the layer, the size
//! of the filter associated to the layer, the stride that is used to slide the filter along
//! the input, as well as the padding that can potentially be applied to the input before application
//! of the convolution.
//!
//! It is assumed that everything (dimensions of the input and the filter as well as stride step and padding)
//! is symmetric in the `x` and `y` directions. If this is not the case, the program can be run twice by specifying
//! the different parameters corresponding to the horizontal and vertical directions separately.
use structopt::StructOpt;

#[derive(Debug, StructOpt)]
/// Compute the dimension of the output of a convolutional layer.
///
/// It is assumed that the input has squared dimension `in_dim`.
/// If this is not the case, the height `h` and width `w` output dimensions
/// can be computed separately by running the program twice and setting
/// `--input-dim h` and `--input-dim w`.
/// The same argument can be made for the filter.
struct Opt {
    #[structopt(short = "i", long = "input-dim")]
    /// The dimension of input.
    in_dim: u16,

    #[structopt(short = "f", long = "filter-size")]
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

    #[structopt(short = "d", long = "deconv")]
    /// Flag that specifies that the layer is deconvolutional.
    deconv: bool
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
        panic!("Filter size ({}) is larger than input ({})!", filter_size, in_dim);
    }
    match repeat {
        0 => in_dim,
        1 => (in_dim - filter_size + 2 * padding) / stride + 1,
        n => conv_output_dim( (in_dim - filter_size + 2 * padding) / stride + 1, filter_size, padding, stride, n - 1)
    }
    
}

/// ## Compute the output dimension of a deconvolutional layer.
///
/// The dimension of the output (o) of the deconvolutional layer is computed from
/// its input dimension `in_dim` (n), the size of its filter `filter_size` (f) as well
/// as the zero-`padding` applied to the input and the `stride` that is used to slide
/// the filter according to:
///
/// o = (n - 1) * s + f - 2*p 
///
/// ## Example
///
/// ```rust
/// assert_eq!(conv_output_dim(28, 5, 0, 1, 1), 24);
/// ```
fn deconv_output_dim(in_dim: u16, filter_size: u16, padding: u16, stride: u16, repeat: u16) -> u16 {
    if filter_size > in_dim + 2 * padding {
        panic!("Filter size ({}) is larger than input ({})!", filter_size, in_dim);
    }
    match repeat {
        0 => in_dim,
        1 => (in_dim - 1) * stride + filter_size - 2 * padding,
        n => conv_output_dim( (in_dim - 1) * stride + filter_size - 2 * padding, filter_size, padding, stride, n - 1)
    }
    
}

fn main() {
    let opt = Opt::from_args();
    if opt.deconv {
        println!(
            "{}",
            deconv_output_dim(opt.in_dim, opt.filter_size, opt.padding, opt.stride, opt.repeat)
        );
    } else {
        println!(
            "{}",
            conv_output_dim(opt.in_dim, opt.filter_size, opt.padding, opt.stride, opt.repeat)
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
    fn test_deconv_output_dim() {
        assert_eq!(deconv_output_dim(32, 2, 0, 2, 1), 64);
    }

    #[test]
    fn test_conv_deconv_chain() {
        let in_dim = 64;
        let stride = 2;
        let filter_size = 3;
        let padding = 1;

        let conv_out = conv_output_dim(in_dim, filter_size, padding, stride, 1);
        assert_eq!(conv_out, 32);
        let deconv_out = deconv_output_dim(conv_out, filter_size, padding, stride, 1);
        assert_eq!(deconv_out, 63);
    }
}

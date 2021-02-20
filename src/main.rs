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
    #[structopt(short="i", long="input-dim")]
    /// The dimension of input.
    in_dim: u16,

    #[structopt(short="f", long="filter-size")]
    /// The filter size.
    filter_size: u16,

    #[structopt(short="p", long="padding", default_value="0")]
    /// The zero-padding that is used for the filter.
    padding: u16,

    #[structopt(short="s", long="stride", default_value="1")]
    /// The stride that is used for the filter.
    stride: u16
    
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
/// assert_eq!(conv_output_dim(28, 5, 0, 1), 24);
/// ```
fn conv_output_dim(in_dim: u16, filter_size: u16, padding: u16, stride: u16) -> u16 {
    (in_dim - filter_size + 2 * padding) / stride + 1
}


fn main() {
    let opt = Opt::from_args();
    println!("{}", conv_output_dim(opt.in_dim, opt.filter_size, opt.padding, opt.stride));
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv_output_dim() {
        assert_eq!(conv_output_dim(28, 5, 0, 1), 24);
    }
}

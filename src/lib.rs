//! # gsim crate
//! This crate provides gsim simulator of quantum circuit. If you looking for farther information of gsim, see the [paper](https://arxiv.org/abs/2308.01432).
//! 
//! # How to use
//! 1. Make gsim with `make_gsim` with arguments initial state, observable, and gate_hamiltonians.
//! 2. Execute simulation with `run_simulation` with arguments gsim and parameters and gate numbers. 
//! 3. You will get measurement expectation value as an output of `run_simulation`.
//! 
//! See more detailed example in [example code](./fn.run_simulation.html#Examples) of`run_simulation`.

mod gsim;
mod interface;

use nalgebra::DMatrix;
use num::complex::Complex64;

pub use gsim::GSim;
pub use interface::{make_gsim,run_simulation,get_dla_ein_adgaterep,simulate_with_params};
pub type CMatrix2 = DMatrix<Complex64>;

#[cfg(test)]
mod tests {
}
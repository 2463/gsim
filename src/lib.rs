// extern crate tensorflow;
pub mod gsim;

use gsim::GSim;
use nalgebra::DMatrix;
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;

pub fn make_gsim(init_density_matrix:CMatrix2,observable:CMatrix2,parameters:Vec<Complex64>,hamiltonians: Vec<CMatrix2>)->GSim{
    GSim {
        init_density_matrix,
        observable,
        parameters,
        hamiltonians,
        dla: Vec::new(),
        dla_ready:false,
    }
}

#[cfg(test)]
mod tests {
}
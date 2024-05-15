pub mod rep;
pub mod sim;
pub mod dla;

use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

use self::sim::{get_e_out, get_prob};

type CMatrix2 = DMatrix<Complex64>;
type RVec = DVector<f64>;

pub struct GSim {
    pub init_density_matrix:CMatrix2,
    pub observable: CMatrix2,
    pub parameters: Vec<Complex64>,
    pub hamiltonians: Vec<CMatrix2>,
    pub dla: Vec<CMatrix2>,
}

pub trait GenerateDLA{
    fn get_dla(&self)->Vec<CMatrix2>;
}

pub trait Simulate{
    fn simulate(&self)->RVec;
}

impl GenerateDLA for GSim{
    fn get_dla(&self)->Vec<CMatrix2>{
        dla::get_dla(&self.hamiltonians)
    }
}

impl Simulate for GSim {
    fn simulate(&self)->RVec {
        let e_out = get_e_out(
            &self.init_density_matrix,
            &self.parameters,
            &self.hamiltonians,
            &self.dla);
        get_prob(e_out, &self.observable, &self.dla)
    }
}
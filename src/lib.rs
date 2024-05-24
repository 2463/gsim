// extern crate tensorflow;
pub mod gsim;

use gsim::{GSim, GenerateDLA};
use nalgebra::{DMatrix,DVector};
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;
type RVec = DVector<f64>;

pub fn make_gsim(init_density_matrix:CMatrix2,observable:CMatrix2,hamiltonians: Vec<CMatrix2>)->GSim{
    let mut gsim = GSim::new(init_density_matrix, observable, hamiltonians);
    gsim.prepare_dla();
    gsim.prepare_e_in();
    gsim
}

pub fn run_simulation(gsim:&mut GSim,parameters:Vec<Complex64>,gate_hamiltonians:Vec<CMatrix2>)->RVec{
    gsim.set_parameters(parameters);
    gsim.simulate(gate_hamiltonians)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::gsim::dla::test_dla;
    use crate::gsim::sim::test_sim;
    use nalgebra::DVector;
    use rand::Rng;
    
    #[test]
    fn make_gsim_test(){
        const NUMBER_OF_QUBIT:u32 = 1;
        let x = test_sim::make_x_pauli();
        let z = test_sim::make_z_pauli();
        let hamiltonians = vec![x.clone(),z.clone()];
        let gate_hamiltonians = vec![x.clone()];
        let number_of_parameters = gate_hamiltonians.len();
        let init = make_zero_density_mat(NUMBER_OF_QUBIT);
        println!("initial state{:.3}",init);
        test_dla::print_cmatrix_in_python_form(&init);
        let mut gsim = make_gsim(init, x, hamiltonians);
        let parameters = make_1_parameters(number_of_parameters);
        let result = run_simulation(&mut gsim, parameters,gate_hamiltonians);
        println!("result : {:.3}",result);
        assert!(false);
    }

    fn make_1_parameters(number_of_parameters:usize)->Vec<Complex64>{
        let mut parameters = Vec::new();
        for _i in 0..number_of_parameters{
            parameters.push(Complex64::new(1.0,0.0));
        }
        parameters
    }

    fn make_pi_2_parameters(number_of_parameters:usize)->Vec<Complex64>{
        let mut parameters = Vec::new();
        for _i in 0..number_of_parameters{
            parameters.push(Complex64::new(1.57079632679,0.0));
        }
        parameters
    }

    fn make_zero_density_mat(number_of_qubit:u32)->CMatrix2{
        let size = 2i64.pow(number_of_qubit) as usize;
        let mut density = CMatrix2::zeros(size, size);
        density[(size-1,size-1)] = Complex64::new(1.0,0.0);
        density
    }

    fn make_random_parameters(number_of_parameters:usize)->Vec<Complex64>{
        let mut parameters = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..number_of_parameters{
            parameters.push(Complex64::new(rng.gen(), rng.gen()));
        }
        parameters
    }

    fn make_random_hermitian(number_of_qubit:u32)->CMatrix2{
        let size = 2i64.pow(number_of_qubit) as usize;
        test_dla::make_random_hermitian(size,size)
    }

    type CVec = DVector<Complex64>;

    fn make_random_init_density_mat(number_of_qubit:u32)->CMatrix2{
        let size = 2i64.pow(number_of_qubit) as usize;
        let mut state = CVec::zeros(size);
        let mut rng = rand::thread_rng();
        for i in 0..size{
            state[i] = Complex64::new(rng.gen(), rng.gen());
        }
        let normalized = state.normalize();
        &normalized * &normalized.adjoint()
    }
}
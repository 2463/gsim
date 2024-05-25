pub(crate) mod rep;
pub(crate) mod sim;
pub(crate) mod dla;

use core::panic;
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;
type RVec = DVector<f64>;

pub struct GSim {
    init_density_matrix:CMatrix2,
    observable: CMatrix2,
    parameters: Option<Vec<f64>>,
    hamiltonians: Vec<CMatrix2>,
    dla: Option<Vec<CMatrix2>>,
    e_in: Option<RVec>,
}

pub trait GenerateDLA{
    fn prepare_dla(&mut self);
}

impl GSim {
    pub fn new(
        init_density_matrix:CMatrix2,
        observable:CMatrix2,
        hamiltonians:Vec<CMatrix2>
    )->GSim {
        GSim {
            init_density_matrix,
            observable,
            hamiltonians,
            parameters: None,
            dla: None,
            e_in: None,
        }
    }

    pub fn set_parameters(&mut self,parameters:Vec<f64>){
        self.parameters = Some(parameters);
    }

    pub fn set_initial_density_matrix(&mut self,initial_density_matrix:CMatrix2){
        self.init_density_matrix = initial_density_matrix;
        self.e_in = None;
    }

    pub fn set_observable(&mut self,observable:CMatrix2){
        self.observable = observable;
    }

    pub fn get_dla(&self)->Vec<CMatrix2>{
        return self.dla.clone().expect("DLA is not generated yet.")
    }

    pub fn prepare_e_in(&mut self) {
        if self.dla == None{panic!("dla is not prepared.")}
        self.e_in = Some(sim::get_e(&self.init_density_matrix, &self.dla.as_ref().unwrap()));
        // println!("## e_in{}",self.e_in.as_ref().unwrap());
    }

    pub fn simulate(&self,gate_hamiltonians:Vec<CMatrix2>)->f64 {
        let e_out = sim::get_e_out(
            &self.e_in.as_ref().expect("e_in is not ready. use `.prepare_e_in()`"),
            self.parameters.as_ref().expect("parameters is not ready. use `set_parameters(parameters)`"),
            gate_hamiltonians,
            &self.dla.as_ref().expect("dla is not ready. use `.prepare_dla()`"));
        sim::get_prob(e_out, &self.observable, &self.dla.as_ref().unwrap())
    }

    pub(crate) fn get_e_out(&self,gate_hamiltonians:&Vec<CMatrix2>)->RVec{
        sim::get_e_out(
            &self.e_in.as_ref().expect("e_in is not ready. use `.prepare_e_in()`"),
            &self.parameters.as_ref().expect("parameters is not ready. use `set_parameters(parameters)`"),
            gate_hamiltonians.clone(),
            &self.dla.as_ref().expect("dla is not ready. use `.prepare_dla()`"))
    }
}

impl GenerateDLA for GSim{
    fn prepare_dla(&mut self){
        self.dla = Some(dla::get_dla(&self.hamiltonians));
    }
}
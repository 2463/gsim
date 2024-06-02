//! This module can build gsim and execute simulation.
//! 
//! このモジュールは量子回路の gsim を実行します．


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
    parameters_and_gate_numbers: Option<Vec<(f64,usize)>>,
    hamiltonians: Vec<CMatrix2>,
    dla: Option<Vec<CMatrix2>>,
    e_in: Option<RVec>,
    ad_rep_gate_hams: Option<Vec<CMatrix2>>,
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
            parameters_and_gate_numbers: None,
            dla: None,
            e_in: None,
            ad_rep_gate_hams: None,
        }
    }

    pub fn set_params_and_gate_numbers(&mut self,parameters_and_gate_numbers:Vec<(f64,usize)>){
        self.parameters_and_gate_numbers = Some(parameters_and_gate_numbers);
    }

    pub fn change_initial_density_matrix(&mut self,initial_density_matrix:CMatrix2){
        self.init_density_matrix = initial_density_matrix;
        self.e_in = None;
    }

    pub fn change_observable(&mut self,observable:CMatrix2){
        self.observable = observable;
    }

    pub fn set_dla(&mut self, dla: Vec<CMatrix2>){
        self.dla = Some(dla)
    }

    pub fn get_dla(&self)->Vec<CMatrix2>{
        return self.dla.clone().expect("DLA is not generated yet.")
    }

    pub fn prepare_e_in(&mut self) {
        if self.dla == None{panic!("dla is not prepared. use `.prepare_dla()`")}
        self.e_in = Some(sim::get_e(&self.init_density_matrix, &self.dla.as_ref().unwrap()));
    }

    pub fn set_e_in(&mut self, e_in: RVec) {
        self.e_in = Some(e_in)
    }

    pub fn get_e_in(&self)->RVec {
        return self.e_in.as_ref().expect("e_in is not prepared. use `gsim.prepare_e_in()`").clone()
    }

    pub fn prepare_gate_hams(&mut self, gate_hamiltonians:Vec<CMatrix2>) {
        if self.dla == None{panic!("dla is not prepared. use `.prepare_dla()`")}
        self.ad_rep_gate_hams = Some(sim::get_ad_rep_gate_hams(gate_hamiltonians, self.dla.as_ref().unwrap()));
    }

    pub fn get_ad_rep_gate_hams(&self)->Vec<CMatrix2>{
        self.ad_rep_gate_hams.as_ref().expect("ad_rep_gate_hams is not prepared.use `.prepare_gate_hams()`").clone()
    }

    pub fn simulate(&self)->f64 {
        let e_out = sim::get_e_out(
            self.e_in.as_ref().expect("Adjoint representation of initial state is not prepared. use `.prepare_e_in()`"),
            self.parameters_and_gate_numbers.as_ref().expect("Parameters and gate numbers are not ready. use `set_params_and_gate_numbers`."),
            self.ad_rep_gate_hams.as_ref().expect("Adjoint representation of gate hamiltonians are not prepared. Use `prepare_gate_hams`."));
        sim::get_prob(e_out, &self.observable, &self.dla.as_ref().unwrap())
    }

}

impl GenerateDLA for GSim{
    fn prepare_dla(&mut self){
        self.dla = Some(dla::get_dla(&self.hamiltonians));
    }
}

pub fn simulate_with_parameters(
    dla: &Vec<CMatrix2>,
    e_in: &RVec,
    observable: &CMatrix2,
    ad_reped_gate_hams: &Vec<CMatrix2>,
    params_and_gate_nums: &Vec<(f64,usize)>
)->f64{
    let e_out = sim::get_e_out(e_in, params_and_gate_nums, ad_reped_gate_hams);
    sim::get_prob(e_out, observable, dla)
}
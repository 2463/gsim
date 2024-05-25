// extern crate tensorflow;
pub mod gsim;

use gsim::{GSim, GenerateDLA};
use nalgebra::DMatrix;
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;
pub fn make_gsim(init_density_matrix:CMatrix2,observable:CMatrix2,gate_hamiltonians: Vec<CMatrix2>)->GSim{
    let mut total_hamiltonians = vec![observable.clone()];
    total_hamiltonians.extend(gate_hamiltonians.clone());
    let mut gsim = GSim::new(init_density_matrix, observable, total_hamiltonians);
    gsim.prepare_dla();
    gsim.prepare_e_in();
    gsim.prepare_gate_hams(gate_hamiltonians);
    gsim
}

pub fn run_simulation(gsim:&mut GSim,parameters_and_gate_number:Vec<(f64,usize)>)->f64{
    gsim.set_params_and_gate_numbers(parameters_and_gate_number);
    gsim.simulate()
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::gsim::dla::test_dla;
    use crate::gsim::sim::test_sim;
    use nalgebra::DVector;
    use rand::Rng;

    type RVec = DVector<f64>;

    const NUMBER_OF_QUBIT : u32 = 5;
    const NUMBER_OF_GATES : u32 = 10;
    const MINUS_I: Complex64 = Complex64::new(0.0,-1.0);
    
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
        let parameters = make_1_parameters_and_gate_numbers(number_of_parameters);
        let result = run_simulation(&mut gsim, parameters);
        println!("result : {:.3}",result);
        assert!(result == 0.00);
    }

    fn make_1_parameters_and_gate_numbers(number_of_parameters:usize)->Vec<(f64,usize)>{
        let mut parameters = Vec::new();
        for i in 0..number_of_parameters{
            parameters.push((1.0,i));
        }
        parameters
    }

    // fn make_pi_2_parameters(number_of_parameters:usize)->Vec<Complex64>{
    //     let mut parameters = Vec::new();
    //     for _i in 0..number_of_parameters{
    //         parameters.push(Complex64::new(1.57079632679,0.0));
    //     }
    //     parameters
    // }

    fn make_zero_density_mat(number_of_qubit:u32)->CMatrix2{
        let size = 2i64.pow(number_of_qubit) as usize;
        let mut density = CMatrix2::zeros(size, size);
        density[(size-1,size-1)] = Complex64::new(1.0,0.0);
        density
    }

    fn make_random_parameters(number_of_parameters:usize)->Vec<(f64,usize)>{
        let mut parameters = Vec::new();
        let mut rng = rand::thread_rng();
        for i in 0..number_of_parameters{
            parameters.push((rng.gen(),i));
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

    fn get_unitary(hamiltonian:&CMatrix2,parameter:f64)->CMatrix2{
        let result = (hamiltonian * Complex64::new(parameter,0.0) * MINUS_I).exp();
        result
    }

    #[test]
    fn make_gsim_random_test(){
        let init = make_random_init_density_mat(NUMBER_OF_QUBIT);
        println!(
            "# initial state\n{:.3}{}\n{}",
            init,
            generate_cmatrix_string_in_python_form(&init),
            generate_cmatrix_rust_form(&init));
        let init_clone = init.clone();
        let mut hams = Vec::new();
        for _i in 0..(NUMBER_OF_GATES){hams.push(make_random_hermitian(NUMBER_OF_QUBIT))}
        for i in 0..NUMBER_OF_GATES{
            println!(
                "## {}-th hamiltonian\n{:.3}{}\n{}",
                i,
                hams[i as usize],
                generate_cmatrix_string_in_python_form(&hams[i as usize]),
                generate_cmatrix_rust_form(&hams[i as usize]),
            )
            };
        let hams_clone = hams.clone();
        let obs = hams[0].clone();
        let obs_clone = hams[0].clone();
        println!("obs{}",obs);
        let paras = make_random_parameters(NUMBER_OF_GATES as usize);
        let paras_clone = paras.clone();
        let mut gsim = make_gsim(init, obs, hams);
        gsim.set_params_and_gate_numbers(paras.clone());
        let result = run_simulation(&mut gsim, paras);
        
        // 通常シミュレーション
        let mut unis = Vec::new();
        for (ham,para) in hams_clone.iter().zip(paras_clone){
            // println!("param:{},hamiltonian{},unitary{}",para,ham,get_unitary(ham, &para));
            unis.push(get_unitary(ham, para.0));
        }
        let mut end_state = init_clone;
        for unitary in unis{
            // println!(
            //     "############# state{}\nunitary{}\n",
            //     end_state,
            //     unitary,
            // );
            end_state = &unitary * &end_state * &unitary.adjoint();
        }
        // println!("end_state{}",end_state);
        let e_out_simulation = get_e_test(&end_state, &gsim.get_dla());
        println!("e_out of normal simulation\n{}",e_out_simulation);

        let normal_result = (&obs_clone * &end_state).trace();

        println!("expectation val(normal): {}",normal_result);
        println!("expectation val (gsim): {}",result);
        let diff = (result - normal_result.re).abs();
        assert!(diff < 1.0e-7,"Difference of gsim and normal is {}",diff);
    }

    fn get_e_test(density_mat:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
        let mut result:RVec = RVec::zeros(gs_dla.len());
        for i in 0..gs_dla.len(){
            result[i] = (&gs_dla[i] * density_mat).trace().im;
        }
        result
    }

        // ####################################### testers ####################################

    // CMatrix2::from_vec(row,col,vec![
    //     Complex::new(0.5, 0.0),
    //     Complex::new(0.5, 0.0)
    // ])
    fn generate_cmatrix_rust_form(mat: &CMatrix2)->String{
        let mut result = String::from("");
        let top1 = "CMatrix2::from_vec(row,col,vec![\n";
        result.push_str(top1);
        for element in mat{
            result.push_str(&format!("            Complex::new({},{}),\n",element.re,element.im));
        }
        let bottom1 = "            ])";
        result.push_str(bottom1);
        result
    }

    fn generate_cmatrix_string_in_python_form(mat: &CMatrix2)->String{
        let mut result = String::from("[");
        for elem in mat.iter(){
            result.push_str(&generate_complex_string_in_python_form(elem));
            result.push_str(",");
        }
        result.push_str("]");
        result
    }

    fn generate_complex_string_in_python_form(c:&Complex64)->String{
        format!("{}+{}j",c.re,c.im)
    }


}
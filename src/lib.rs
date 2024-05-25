// extern crate tensorflow;
pub mod gsim;

use gsim::{GSim, GenerateDLA};
use nalgebra::DMatrix;
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;
pub fn make_gsim(init_density_matrix:CMatrix2,observable:CMatrix2,hamiltonians: &Vec<CMatrix2>)->GSim{
    let mut gsim = GSim::new(init_density_matrix, observable, hamiltonians.clone());
    gsim.prepare_dla();
    gsim.prepare_e_in();
    gsim
}

pub fn run_simulation(gsim:&mut GSim,parameters:Vec<f64>,gate_hamiltonians:Vec<CMatrix2>)->f64{
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
    use num::Complex;

    type RVec = DVector<f64>;

    const NUMBER_OF_QUBIT : u32 = 1;
    const NUMBER_OF_GATES : u32 = 2;
    const I : Complex64 = Complex64::new(0.0,1.0);
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
        let mut gsim = make_gsim(init, x, &hamiltonians);
        let parameters = make_1_parameters(number_of_parameters);
        let result = run_simulation(&mut gsim, parameters,gate_hamiltonians);
        println!("result : {:.3}",result);
        assert!(result == 0.00);
    }

    fn make_1_parameters(number_of_parameters:usize)->Vec<f64>{
        let mut parameters = Vec::new();
        for _i in 0..number_of_parameters{
            parameters.push(1.0);
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

    fn get_unitary(hamiltonian:&CMatrix2,parameter:f64)->CMatrix2{
        let result = (hamiltonian * Complex64::new(parameter,0.0) * MINUS_I).exp();
        result
    }

    #[test]
    fn test_nalgebra(){
        let row = 2;
        let col = 2;
        let mat0 = CMatrix2::from_vec(row,col,vec![
            Complex::new(0.0,0.0),
            Complex::new(0.1,0.0),
            Complex::new(0.2,0.0),
            Complex::new(0.3,0.0),
            ]);
        let mat1 = CMatrix2::from_vec(row,col,vec![
            Complex::new(0.0,0.0),
            Complex::new(0.0,0.1),
            Complex::new(0.0,0.2),
            Complex::new(0.0,0.3),
            ]);

        println!("{}",&mat0 * &mat1 * &mat0.adjoint());
        println!("{}",(mat0 * Complex64::new(0.4, 0.8) * MINUS_I).exp());
        assert!(false);
    }

    #[test]
    fn make_gsim_fixed_test(){
        let row = 2;
        let col = 2;
        let init = CMatrix2::from_vec(row,col,vec![
            Complex::new(0.06780365607161457,0.0),
            Complex::new(0.14774422302913814,0.20341574387557385),
            Complex::new(0.14774422302913814,-0.20341574387557385),
            Complex::new(0.9321963439283855,0.0),
            ]);
        println!("init {}",init);
        let init_clone = init.clone();
        let ham0 = CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.8,0.0),
            Complex::new(0.9,-1.0),
            Complex::new(0.9,1.0),
            Complex::new(0.2,0.0),
            ]);
        let ham1 =CMatrix2::from_vec(row,col,vec![
            Complex::new(0.9,0.0),
            Complex::new(-0.2,-0.6),
            Complex::new(-0.2,0.6),
            Complex::new(-0.2,0.0),
            ]);
        let hams = vec![ham0,ham1];
        let hams_clone = hams.clone();
        let obs = hams[0].clone();
        let obs_clone = hams[0].clone();
        println!("obs{}",obs);
        let paras = vec![0.4,0.1];
        let paras_clone = paras.clone();
        let mut gsim = make_gsim(init, obs, &hams);
        gsim.set_parameters(paras.clone());
        let e_out = gsim.get_e_out(&hams);
        let dla = gsim.get_dla();
        println!("dla");
        for elem in dla{println!("{}",generate_cmatrix_string_in_python_form(&elem))}
        let result = run_simulation(&mut gsim, paras, hams);
        
        // 通常シミュレーション
        let mut unis = Vec::new();
        for (ham,para) in hams_clone.iter().zip(paras_clone){
            // println!("param:{},hamiltonian{},unitary{}",para,ham,get_unitary(ham, &para));
            unis.push(get_unitary(ham, para));
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
        println!("e_out of gsim\n{}",e_out);

        let normal_result = (&obs_clone * &end_state).trace();

        println!("expectation val(normal): {}",normal_result);
        println!("expectation val (gsim): {}",result);
        assert!(false);
        
    }


    // #[test]
    // fn make_gsim_random_test(){
    //     let init = make_random_init_density_mat(NUMBER_OF_QUBIT);
    //     println!(
    //         "# initial state\n{:.3}{}\n{}",
    //         init,
    //         generate_cmatrix_string_in_python_form(&init),
    //         generate_cmatrix_rust_form(&init));
    //     let init_clone = init.clone();
    //     let mut hamiltonians = Vec::new();
    //     for _i in 0..(NUMBER_OF_GATES){hamiltonians.push(make_random_hermitian(NUMBER_OF_QUBIT))}
    //     for i in 0..NUMBER_OF_GATES{
    //         println!(
    //             "## {}-th hamiltonian\n{:.3}{}\n{}",
    //             i,
    //             hamiltonians[i as usize],
    //             generate_cmatrix_string_in_python_form(&hamiltonians[i as usize]),
    //             generate_cmatrix_rust_form(&hamiltonians[i as usize]),
    //         )
    //         };
    //     let hamiltonians_clone = hamiltonians.clone();
    //     let observable = hamiltonians[(NUMBER_OF_GATES - 1) as usize].clone();
    //     let observable_clone = observable.clone();
    //     let parameters = make_random_parameters((NUMBER_OF_GATES) as usize);
    //     let parameters_clone = parameters.clone();
    //     print!("parameters [");
    //     for i in 0..NUMBER_OF_GATES{print!("{}, ",generate_complex_string_in_python_form(&parameters[i as usize]))}
    //     println!("]");
    //     let mut gsim = make_gsim(init, observable, &hamiltonians);
    //     gsim.set_parameters(parameters.clone());
    //     let e_out = gsim.get_e_out(&hamiltonians);
    //     println!("e_out of gsim\n{}",e_out);
    //     let result = run_simulation(&mut gsim, parameters, hamiltonians);
    //     // 通常のシミュレーション
    //     let mut unitaries = Vec::new();
    //     for (ham,param) in hamiltonians_clone.iter().zip(parameters_clone){
    //         println!("param:{},hamiltonian{},unitary{}",param,ham,get_unitary(ham, &param));
    //         unitaries.push(get_unitary(ham, &param));
    //     }
    //     for i in 0..NUMBER_OF_GATES{println!("## {}-th unitary\n{:.3}{}",i,unitaries[i as usize],generate_cmatrix_string_in_python_form(&unitaries[i as usize]))};
    //     let mut end_state = init_clone;
    //     for unitary in unitaries{
    //         end_state = unitary.conjugate().transpose() * end_state * unitary;
    //     }
    //     println!("end_state{}",end_state);
    //     let measurement_result = (observable_clone * &end_state).trace().re;
    //     println!("gsim result {}\nmeasurement_result {}",result,measurement_result);
    //     let e_out_simulation = get_e_test(&end_state, &gsim.get_dla());
    //     println!("e_out of normal simulation\n{}",e_out_simulation);
    //     // assert!(result == measurement_result);
    //     assert!(true);

    // }

    fn get_e_test(density_mat:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
        let mut result:RVec = RVec::zeros(gs_dla.len());
        for i in 0..gs_dla.len(){
            result[i] = (&gs_dla[i] * density_mat).trace().im;
        }
        result
    }

        // ####################################### testers ####################################

    // let mut vector_of_cmat: Vec<CMatrix2> = Vec::new();
    // let row = 2;
    // let col = 2;
    // vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
    //     Complex::new(0.5, 0.0),
    //     Complex::new(0.5, 0.0)
    // ]));
    pub fn generate_dla_rust_form(row:usize,col:usize,matrices: &Vec<CMatrix2>)->String{
        let mut result = String::from("");
        let first_line = "    let mut vector_of_cmat: Vec<CMatrix2> = Vec::new();\n";
        let second_line = format!("   let row = {};\n",row);
        let third_line = format!("    let col = {};\n",col);
        result.push_str(&first_line);
        result.push_str(&second_line);
        result.push_str(&third_line);
        for mat in matrices{
            let top = "        vector_of_cmat.push(";
            result.push_str(top);
            let matrix_str = generate_cmatrix_rust_form(mat);
            result.push_str(&matrix_str);
            let bottom = ");\n";
            result.push_str(bottom);
        }
        result
    }

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
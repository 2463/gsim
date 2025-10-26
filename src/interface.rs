use crate::gsim;

use super::gsim::{GSim,GenerateDLA};
use nalgebra::{DMatrix, DVector};
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;


/// Make gsim simulator.
/// # Setting Up `gate_hamiltonians`
/// - The `gate_hamiltonians` must be defined as a global Hamiltonian.
/// - Let's consider an example. If we aim to apply the `exp(-i pi / 2 x)` gate to the first qubit in a three-qubit circuit,
///  we need to substitute `X I I` Hamiltonian for the `gate_hamiltonians`.
/// # Examples
/// Following example makes gsim simulator which simulate a circuit that can use Pauli X and Pauli Z hamiltonian as gates with Observable X.
/// 
/// ```
/// use nalgebra::DMatrix;
/// use num::complex::Complex64;
/// 
/// type CMatrix = DMatrix<Complex64>;
/// 
/// const ONE: Complex64 = Complex64::new(1.0,0.0);
/// const MINUS_ONE: Complex64 = Complex64::new(-1.0,0.0);
/// const ZERO: Complex64 = Complex64::new(0.0,0.0);
/// 
/// let init_state = DMatrix::from_vec(2,2,vec![ZERO,ZERO,ZERO,ONE]);
/// let pauli_x = CMatrix::from_vec(2,2,vec![ZERO,ONE,ONE,ZERO]);
/// let pauli_z = CMatrix::from_vec(2,2,vec![ONE,ZERO,ZERO,MINUS_ONE]);
/// let pauli_z_obs = pauli_z.clone();
/// let gsim = gsim::make_gsim(
///     init_state,
///     pauli_z_obs,
///     vec![pauli_x,pauli_z],     
/// );
/// ```
pub fn make_gsim<'a>(init_density_matrix:CMatrix2,observable:CMatrix2,gate_hamiltonians: Vec<CMatrix2>)->GSim{
    let mut total_hamiltonians = vec![observable.clone()];
    total_hamiltonians.extend(gate_hamiltonians.clone());
    let mut gsim = GSim::new(init_density_matrix, observable, total_hamiltonians);
    gsim.prepare_dla();
    gsim.prepare_e_in();
    gsim.prepare_gate_hams(gate_hamiltonians);
    gsim
}

/// Execute simulation with gsim simulator, given gate numbers, and parameters.
/// 
/// # Determining `parameters_and_gate_numbers`
/// - `parameters_and_gate_numbers` is a vector that contains tuples. Each tuple consists of a `parameter` and a `gate_number`, arranged in the sequence they are applied to the circuit.
/// - The `parameter` is simply a value of type f64.
/// - The `gate_number` is the assigned number of the Hamiltonian you wish to apply in your circuit. 
/// This number corresponds to the order of the Hamiltonian in the `gate_hamiltonians` declared in the argument of `make_gsim`.
/// # Examples
/// - Let's assume created `gsim` with `gate_hamiltonians=vec![pauli_X,pauli_Z]`. 
/// - If we want to apply the first gate with the Hamiltonian `pauli_Z` and a `parameter` of 1.57079632679, 
/// then `parameters_and_gate_numbers` should be `vec![(1.57079632679,1)]`.
/// - Here, `1.57079632679` is the parameter and `1` represents the position of `pauli_Z` in `gate_hamiltonians`.
/// 
/// (initial state = |0>, obs = Z, unitary = exp(-i * pi/2 * Z)),
/// ```
/// use nalgebra::DMatrix;
/// use num::complex::Complex64;
/// 
/// type CMatrix = DMatrix<Complex64>;
/// 
/// const ONE: Complex64 = Complex64::new(1.0,0.0);
/// const MINUS_ONE: Complex64 = Complex64::new(-1.0,0.0);
/// const ZERO: Complex64 = Complex64::new(0.0,0.0);
/// 
/// let init_state = DMatrix::from_vec(2,2,vec![ZERO,ZERO,ZERO,ONE]);
/// let pauli_x = CMatrix::from_vec(2,2,vec![ZERO,ONE,ONE,ZERO]);
/// let pauli_z = CMatrix::from_vec(2,2,vec![ONE,ZERO,ZERO,MINUS_ONE]);
/// let pauli_z_obs = pauli_z.clone();
/// let mut gsim = gsim::make_gsim(
///     init_state,
///     pauli_z_obs,
///     vec![pauli_x,pauli_z],     
/// );
/// let result = gsim::run_simulation(
///     &mut gsim,
///     vec![(1.57079632679,1)],
/// );
/// assert!((result - (-1.0)).abs() < 1.0e-12);
/// ```
/// - If you want to apply many gates, do as follows.
/// - Following circuit implies (param, Hamiltonian), (0.5, Z), (1.0, X), (0.1, X).
/// ```
/// /// use nalgebra::DMatrix;
/// use nalgebra::DMatrix;
/// use num::complex::Complex64;
/// 
/// type CMatrix = DMatrix<Complex64>;
/// 
/// const ONE: Complex64 = Complex64::new(1.0,0.0);
/// const MINUS_ONE: Complex64 = Complex64::new(-1.0,0.0);
/// const ZERO: Complex64 = Complex64::new(0.0,0.0);
/// 
/// let init_state = DMatrix::from_vec(2,2,vec![ZERO,ZERO,ZERO,ONE]);
/// let pauli_x = CMatrix::from_vec(2,2,vec![ZERO,ONE,ONE,ZERO]);
/// let pauli_z = CMatrix::from_vec(2,2,vec![ONE,ZERO,ZERO,MINUS_ONE]);
/// let pauli_z_obs = pauli_z.clone();
/// let mut gsim = gsim::make_gsim(
///     init_state,
///     pauli_z_obs,
///     vec![pauli_x,pauli_z],     
/// );
/// let result = gsim::run_simulation(
///     &mut gsim,
///     vec![(0.5,1),(1.0,0),(0.1,0)],
/// );
/// assert!((result - (0.588501117255346)).abs() < 1.0e-12);
/// ```
pub fn run_simulation(gsim:&mut GSim,parameters_and_gate_numbers:Vec<(f64,usize)>)->f64{
    gsim.set_params_and_gate_numbers(parameters_and_gate_numbers);
    gsim.simulate()
}

pub fn get_dla_ein_adgaterep<'a>(init_density_matrix:CMatrix2,observable:CMatrix2,gate_hamiltonians: Vec<CMatrix2>)->(Vec<CMatrix2>,DVector<f64>,Vec<CMatrix2>){
    let dla = make_gsim(init_density_matrix, observable, gate_hamiltonians);
    return (dla.get_dla(),dla.get_e_in(),dla.get_ad_rep_gate_hams())
}

pub fn get_dla(vector_of_hams: &Vec<CMatrix2>)->Vec<CMatrix2>{
    gsim::dla::get_dla(&vector_of_hams)
}

pub fn get_e_in(init: &CMatrix2, dla:&Vec<CMatrix2>)->DVector<f64>{
    gsim::sim::get_e(init,dla)
}

pub fn get_ad_rep_gate_hams(gate_hamiltonians: Vec<CMatrix2>,dla: &Vec<CMatrix2>)->Vec<CMatrix2>{
    gsim::sim::get_ad_rep_gate_hams(gate_hamiltonians,dla)
}

pub fn simulate_with_params(
    dla: &Vec<CMatrix2>,
    e_in: &DVector<f64>,
    ad_reped_gate_hams: &Vec<CMatrix2>,
    observable: &CMatrix2,
    params_and_gate_nums: &Vec<(f64,usize)>
)->f64{
    gsim::simulate_with_parameters(dla, e_in, observable, ad_reped_gate_hams, params_and_gate_nums)
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::gsim::dla::test_dla;
    use nalgebra::DVector;
    use rand::Rng;

    type RVec = DVector<f64>;

    const NUMBER_OF_QUBIT : u32 = 1;
    const NUMBER_OF_GATES : u32 = 1;
    const MINUS_I: Complex64 = Complex64::new(0.0,-1.0);
    
    fn make_random_parameters(number_of_parameters:usize)->Vec<(f64,usize)>{
        let mut parameters = Vec::new();
        let mut rng = rand::thread_rng();
        for _i in 0..number_of_parameters{
            let gate_number : usize = ((number_of_parameters as f64) * rng.gen::<f64>()) as usize;
            println!("gate_number {}",gate_number);
            parameters.push((rng.gen(),gate_number));
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
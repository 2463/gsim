use super::rep;
use nalgebra::{DMatrix, DVector};
use num::complex::{Complex64, ComplexFloat};
use itertools;

type CMatrix2 = DMatrix<Complex64>;
type Matrix2 = DMatrix<f64>;
type RVec = DVector<f64>;

// 1. unitary を得る
// 2. $e_in$ を作る．つまり，G の G_\alpha についての，初期状態 $\rho_in$ の測定期待値のベクトル $e_in$ を作成する
// 3. obs を g の g の和によって表現する（ためのベクトル $w$ を得る : $O=\sum_\alpha w_\alpha G_\alpha$）

// 1.
fn get_unitary(hamiltonian:&CMatrix2,parameter:&Complex64)->Matrix2{
    let result = (hamiltonian * *parameter * Complex64::new(0.0, -1.0)).exp();
    cast_in_real(&result)
}

fn cast_in_real(cmat:&CMatrix2)->Matrix2{
    let mut result = Matrix2::zeros(cmat.nrows(), cmat.ncols());
    for (i,j) in itertools::iproduct!(0..cmat.ncols(),0..cmat.nrows()){
        result[(i,j)] = cmat[(i,j)].re
    }
    result
}

// 2.
pub(super) fn get_e_in(init_density_mat:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut result:RVec = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
        println!(
            "# e_in elemnt in get_e_in\n## dla{}\n## density mat{}## trace {}\n",
            gs_dla[i],
            init_density_mat,
            (&gs_dla[i] * init_density_mat).trace());
        result[i] = (&gs_dla[i] * init_density_mat).trace().im();
    }
    result
}

// 3.
fn decompose_obs(observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut w = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
        w[i] = -observable.dot(&gs_dla[i]).re();
    }
    w
}

pub(super) fn get_e_out(e_in:&RVec,parameters:&Vec<Complex64>,hamiltonians:Vec<CMatrix2>,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut rep_unitaries = Vec::new();
    for (parameter, hamiltonian) in parameters.iter().zip(hamiltonians.iter()){
        // println!(
        //     "param {}\nhamiltonian{:.3}\nunitary{:.3}\nrep_unitary{:.3}",
        //     parameter,
        //     hamiltonian,
        //     get_unitary(hamiltonian, parameter),
        //     rep::adjoint_rep(get_unitary(hamiltonian, parameter),gs_dla));
        rep_unitaries.push(get_unitary(&rep::adjoint_rep(hamiltonian,gs_dla), parameter));
    }

    let mut e_out = e_in.clone();

    println!("e_in{:.3}",e_in);
    println!("e_out = unitary * e_in");
    println!("hamiltonian{:.3},unitary{:.3}",hamiltonians[0],get_unitary(&hamiltonians[0], &Complex64::new(1.57079632679,0.0)));
    println!("rep_unitary{:.3}",rep_unitaries[0]);

    for unitary in rep_unitaries{
        e_out = unitary * e_in;
    }

    e_out
}

pub(super) fn get_prob(e_out:RVec,observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let w = decompose_obs(observable, gs_dla);
    println!("(in get_prob)w : {:.3}\ne_out: {:.3}",w,e_out);
    w * e_out
}

#[cfg(test)]
pub mod test_sim{
    use super::*;

    pub fn make_x_pauli()->CMatrix2{
        let mut x = CMatrix2::zeros(2, 2);
        x[(0,1)] = Complex64::new(1.0,0.0);
        x[(1,0)] = Complex64::new(1.0,0.0);
        x
    }

    pub fn make_z_pauli()->CMatrix2{
        let mut z = CMatrix2::zeros(2, 2);
        z[(0,0)] = Complex64::new(1.0, 0.0);
        z[(1,1)] = Complex64::new(-1.0,0.0);
        z
    }

    fn is_close(c:CMatrix2,d:CMatrix2)->bool{
        (&c - d).norm() < 1.0e-7 * (c.ncols() * c.nrows() * 2) as f64
    }
    #[test]
    fn get_e_in_test(){
        
    }
}
use super::rep;
use nalgebra::{DMatrix, DVector};
use num::complex::{Complex64, ComplexFloat};
use itertools;

type CMatrix2 = DMatrix<Complex64>;
type Matrix2 = DMatrix<f64>;
type RVec = DVector<f64>;

const I : Complex64 = Complex64::new(0.0,1.0);
const MINUS_I : Complex64 = Complex64::new(0.0,-1.0);

// 1. unitary を得る
// 2. $e_in$ を作る．つまり，G の G_\alpha についての，初期状態 $\rho_in$ の測定期待値のベクトル $e_in$ を作成する
// 3. obs を g の g の和によって表現する（ためのベクトル $w$ を得る : $O=\sum_\alpha w_\alpha G_\alpha$）

// 1.
fn get_unitary(hamiltonian:&CMatrix2,parameter:f64)->Matrix2{
    let result = (hamiltonian * Complex64::new(parameter,0.0) * MINUS_I).exp();
    // println!("## get_unitary\nresult{:.3}",result);
    cast_in_real(&result)
}

fn cast_in_real2(cmat:&CMatrix2)->CMatrix2{
    let mut result = CMatrix2::zeros(cmat.nrows(), cmat.ncols());
    for (i,j) in itertools::iproduct!(0..cmat.ncols(),0..cmat.nrows()){
        result[(i,j)] = Complex64::new(cmat[(i,j)].im,0.0);
    }
    result
}

fn cast_in_real(cmat:&CMatrix2)->Matrix2{
    let mut result = Matrix2::zeros(cmat.nrows(), cmat.ncols());
    for (i,j) in itertools::iproduct!(0..cmat.ncols(),0..cmat.nrows()){
        result[(i,j)] = cmat[(i,j)].re
    }
    result
}

// 2.
pub(super) fn get_e(density_mat:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut result:RVec = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
        // println!(
        //     "# e_in elemnt in get_e_in\n## dla{}\n## density mat{}## trace {}\n",
        //     gs_dla[i],
        //     density_mat,
        //     (&gs_dla[i] * density_mat).trace());
        result[i] = (&gs_dla[i] * density_mat).trace().im();
    }
    result
}

// 3.
fn decompose_obs(observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut w = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
        w[i] = rep::ip(observable,&gs_dla[i]).im();
    }
    println!("w :\n{}",w);
    w
}

pub(super) fn get_e_out(e_in:&RVec,parameters:&Vec<f64>,hamiltonians:Vec<CMatrix2>,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut rep_unitaries = Vec::new();
    for (parameter, hamiltonian) in parameters.iter().zip(hamiltonians.iter()){
        let ad_ham = rep::adjoint_rep(hamiltonian, gs_dla);
        let ad_u = get_unitary(&ad_ham, *parameter);
        // println!(
        //     "# param {}\n # hamiltonian{:.3}\n# unitary{:.3}\n# ad_ham{:.3}\n# parameter{}\n# ad_u{:.3}",
        //     parameter,
        //     hamiltonian,
        //     get_unitary(hamiltonian, parameter),
        //     ad_ham,
        //     parameter,
        //     ad_u
        // );
        rep_unitaries.push(ad_u);
    }

    let mut e_out = e_in.clone();

    for unitary in rep_unitaries{
        e_out = unitary * e_out;
    }

    e_out
}

pub(super) fn get_prob(e_out:RVec,observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->f64{
    let w = decompose_obs(observable, gs_dla);
    println!("(in get_prob)w : {:.3}\ne_out: {:.3}",w,e_out);
    (w.transpose() * &e_out)[0]
}

#[cfg(test)]
pub mod test_sim{
    use super::*;

    pub(crate) fn make_x_pauli()->CMatrix2{
        let mut x = CMatrix2::zeros(2, 2);
        x[(0,1)] = Complex64::new(1.0,0.0);
        x[(1,0)] = Complex64::new(1.0,0.0);
        x
    }

    pub(crate) fn make_z_pauli()->CMatrix2{
        let mut z = CMatrix2::zeros(2, 2);
        z[(0,0)] = Complex64::new(1.0, 0.0);
        z[(1,1)] = Complex64::new(-1.0,0.0);
        z
    }
}
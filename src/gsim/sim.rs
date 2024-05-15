use super::rep;
use nalgebra::{DMatrix, DVector};
use num::complex::{Complex64, ComplexFloat};

type CMatrix2 = DMatrix<Complex64>;
type RVec = DVector<f64>;

// 1. unitary を得る
// 2. $e_in$ を作る．つまり，G の G_\alpha についての，初期状態 $\rho_in$ の測定期待値のベクトル $e_in$ を作成する
// 3. obs を g の g の和によって表現する（ためのベクトル $w$ を得る : $O=\sum_\alpha w_\alpha G_\alpha$）

// 1.
fn get_unitary(hamiltonian:&CMatrix2,parameter:&Complex64)->CMatrix2{
    (hamiltonian * *parameter * Complex64::new(0.0, 1.0)).exp()
}

// 2.
fn get_e_in(init_density_mat:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let mut result:RVec = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
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

pub(super) fn get_e_out(init_density_mat:&CMatrix2,parameters:&Vec<Complex64>,hamiltonians:&Vec<CMatrix2>,gs_dla:&Vec<CMatrix2>)->RVec{
    let e_in = get_e_in(&init_density_mat, gs_dla);
    let mut rep_unitaries = Vec::new();
    for (parameter, hamiltonian) in parameters.iter().zip(hamiltonians.iter()){
        rep_unitaries.push(rep::adjoint_rep(get_unitary(hamiltonian, parameter),gs_dla));
    }

    let mut e_out = e_in.clone();

    for unitary in rep_unitaries{
        e_out = unitary * &e_in;
    }

    e_out
}

pub(super) fn get_prob(e_out:RVec,observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->RVec{
    let w = decompose_obs(observable, gs_dla);
    w * e_out
}

#[cfg(test)]
mod test_sim{
    use super::*;

    fn make_x_pauli()->CMatrix2{
        let mut x = CMatrix2::zeros(2, 2);
        x[(0,1)] = Complex64::new(1.0,0.0);
        x[(1,0)] = Complex64::new(1.0,0.0);
        x
    }

    fn is_close(c:CMatrix2,d:CMatrix2)->bool{
        (&c - d).norm() < 1.0e-7 * (c.ncols() * c.nrows() * 2) as f64
    }

    #[test]
    fn get_unitary_test(){
        let h = make_x_pauli();
        let param = Complex64::new(1.57079632679,0.0); // 1.57... = pi/2
        let u = get_unitary(&h, &param);
        let accurate_u = make_x_pauli() * Complex64::new(0.0,1.0);
        println!("param : {}\nh{}u{}",param,h,u);
        assert!(is_close(u, accurate_u));
    }

    #[test]
    fn get_e_in_test(){
        
    }
}
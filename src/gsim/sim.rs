use super::rep;
use nalgebra::{DMatrix, DVector};
use num::complex::{Complex64, ComplexFloat};
use indicatif::{ProgressBar, ProgressStyle};
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
    cast_in_real(result)
}

fn cast_in_real(cmat:CMatrix2)->Matrix2{
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
    let imaginalized_obs = observable * I;
    let mut w = RVec::zeros(gs_dla.len());
    for i in 0..gs_dla.len(){
        w[i] = rep::ip(&imaginalized_obs,&gs_dla[i]).re; //歪エルミート同士をかけあわせている
    }
    // println!("w :\n{}",w);
    w
}

pub(super) fn get_ad_rep_gate_hams(gate_hams:Vec<CMatrix2>,dla:&Vec<CMatrix2>)->Vec<CMatrix2>{
    // (get adjoint represented gate hamiltonians)
    let pb = ProgressBar::new(gate_hams.len() as u64);
    let bar_style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})\n{msg}"
    )
    .unwrap();
    pb.set_style(bar_style);
    pb.set_message("Making adjoint representation of gate hamiltonians...");

    let mut result :Vec<CMatrix2> = Vec::new();
    for ham in gate_hams{
        pb.inc(1);
        let ad_ham = rep::adjoint_rep(ham, dla);
        result.push(ad_ham);
    }
    result
}

pub(super) fn get_e_out(e_in:&RVec,params_and_gate_nums:&Vec<(f64,usize)>,ad_reped_gate_hams:&Vec<CMatrix2>)->RVec{
    let pb = ProgressBar::new(params_and_gate_nums.len() as u64);
    let bar_style = ProgressStyle::with_template(
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})\n{msg}"
    )
    .unwrap();
    pb.set_style(bar_style);
    pb.set_message("Simulating with gsim...");

    let mut e_out = e_in.clone();
    for (param,number) in params_and_gate_nums{
        pb.inc(1);
        let ad_u = get_unitary(&ad_reped_gate_hams[*number], *param);
        e_out = ad_u * e_out;
    }
    e_out
}

// pub(super) fn get_e_out_dep(e_in:&RVec,parameters:&Vec<f64>,hamiltonians:Vec<CMatrix2>,gs_dla:&Vec<CMatrix2>)->RVec{
//     let mut rep_unitaries = Vec::new();
//     let pb = ProgressBar::new(parameters.len() as u64);
//     let bar_style = ProgressStyle::with_template(
//         "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})\n{msg}"
//     )
//     .unwrap();
//     pb.set_style(bar_style);
//     pb.set_message("Simulating with gsim...");

//     for (parameter, hamiltonian) in parameters.iter().zip(hamiltonians.iter()){
//         pb.inc(1);
//         let ad_ham = rep::adjoint_rep(hamiltonian, gs_dla);
//         let ad_u = get_unitary(&ad_ham, *parameter);
//         // println!(
//         //     "# param {}\n # hamiltonian{:.3}\n# unitary{:.3}\n# ad_ham{:.3}\n# parameter{}\n# ad_u{:.3}",
//         //     parameter,
//         //     hamiltonian,
//         //     get_unitary(hamiltonian, parameter),
//         //     ad_ham,
//         //     parameter,
//         //     ad_u
//         // );
//         rep_unitaries.push(ad_u);
//     }

//     let mut e_out = e_in.clone();

//     for unitary in rep_unitaries{
//         e_out = unitary * e_out;
//     }

//     e_out
// }

pub(super) fn get_prob(e_out:RVec,observable:&CMatrix2,gs_dla:&Vec<CMatrix2>)->f64{
    let w = decompose_obs(observable, gs_dla);
    // println!("(in get_prob)w : {:.3}\ne_out: {:.3}",w,e_out);
    (w.transpose() * &e_out)[0]
}

#[cfg(test)]
pub mod test_sim{
    // use super::*;

    // pub(crate) fn make_x_pauli()->CMatrix2{
    //     let mut x = CMatrix2::zeros(2, 2);
    //     x[(0,1)] = Complex64::new(1.0,0.0);
    //     x[(1,0)] = Complex64::new(1.0,0.0);
    //     x
    // }

    // pub(crate) fn make_z_pauli()->CMatrix2{
    //     let mut z = CMatrix2::zeros(2, 2);
    //     z[(0,0)] = Complex64::new(1.0, 0.0);
    //     z[(1,1)] = Complex64::new(-1.0,0.0);
    //     z
    // }
}
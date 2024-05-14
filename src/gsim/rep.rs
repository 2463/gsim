use crate::gsim::dla::{self, is_zero};
use super::dla::commutator;
use nalgebra::DMatrix;
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;

// DLA からシュミット直交基底を得る
// 共役表現でハミルトニアンを低次元に写像する

pub fn get_schmit_basis(mut dla: Vec<CMatrix2>)->Vec<CMatrix2>{
    let first_element = dla.pop().expect("The vector is empty");

    let mut sch_basis: Vec<CMatrix2> = vec![first_element];
    while let Some(element) = dla.pop() {
        let new_base = gs_system(element, &sch_basis);
        let new_base = smallize(&new_base);
        sch_basis.push(new_base);
    }
    sch_basis
}

pub fn adjoint_rep(target:CMatrix2,sch_basis: &Vec<CMatrix2>)->CMatrix2{
    let dim = sch_basis.len();
    let mut rep = DMatrix::zeros(dim,dim);
    for (i,j) in itertools::iproduct!(0..dim, 0..dim){
        let value = &target * commutator(&sch_basis[i], &sch_basis[j]);
        rep[(i,j)] = value.trace();
    }
    rep
}

// // c_n - \sum_i (<c_i,c_n>/<c_i,c_i>)c_i を計算する場所
// pub(super) fn make_new_base(target: CMatrix2,basis: &Vec<CMatrix2>)->CMatrix2{
//     let mut result = target.clone();

//     for base in basis{
//         let coef = base.dot(&target) / base.dot(&base);
//         result = result - (base * coef);
//     }

//     if dla::is_zero(&result){
//         return result;
//     }
    
//     // normalize はよくない?
//     normalize(&result)
// }

pub fn gs_system(target: CMatrix2,system: &Vec<CMatrix2>)->CMatrix2{
    if is_zero(&target){
        return CMatrix2::zeros(target.nrows(), target.ncols());
    }
    let mut gs = target;
    // println!("gs [input]: {}", generate_cmatrix_string_in_python_form(&gs));
        for base in system{
        gs = gram_schmidt(gs, base);
        if dla::is_zero(&gs){
            // println!("gs : return zero");
            return CMatrix2::zeros(gs.nrows(), gs.ncols());
        }    
        // println!("gs : {}", generate_cmatrix_string_in_python_form(&gs));
    }
    // println!("gs [return]: {}",generate_cmatrix_string_in_python_form(&normalize(&gs)));
    gs
}
fn gram_schmidt(target: CMatrix2,base: &CMatrix2)->CMatrix2{
    let result = &target - base * (base.dot(&target) / base.dot(&base));
    result
}

pub fn normalize(c:&CMatrix2)->CMatrix2{
    // ここのしきい値を動的に変えるシステムを考える．（e-4 とかじゃないと0以外を）
    // if c.norm() < 1.0e-9 * (c.nrows() * c.ncols() * 2) as f64{
    //     return CMatrix2::zeros(c.nrows(), c.ncols());
    // }
    c * Complex64::new(1.0 / c.norm(),0.0)
}

pub fn smallize(c:&CMatrix2)->CMatrix2{
    // 1より大きい場合のみ小さくする
    if c.norm() < 1.0{
        return c.clone();
    }
    c * Complex64::new(1.0 / c.norm(),0.0)
}

#[cfg(test)]
pub mod test_rep{
    use super::super::dla::test_dla::*;
    use itertools::Itertools;
    use num::{complex::ComplexFloat, pow::Pow, Complex};

    use super::*;

    pub fn check_linear_ind_intest(matrix_a : &CMatrix2, matrix_b : &CMatrix2)->bool{
        (matrix_a.dot(matrix_a) * matrix_b.dot(matrix_b) - matrix_a.dot(matrix_b).powi(2)).abs() > 1.0e-7
    }


    // fn make_minus(target:&CMatrix2,base:&CMatrix2)->CMatrix2{
    //     let mut minus: CMatrix2 = DMatrix::zeros(target.ncols(),target.nrows());
    
    //     let coef = base.dot(&target) / base.dot(&base);
    //     minus = (base * coef) + minus;
    
    //     minus
    // }    

    fn check_hermitian(matrix: &CMatrix2) -> bool {
        let (rows, cols) = (matrix.ncols(),matrix.nrows());
        for i in 0..rows {
            for j in 0..cols {
                if matrix[(i, j)] != matrix[(j, i)].conj() {
                    return false;
                }
            }
        }
        true
    }

    pub fn normalize_all(input : Vec<CMatrix2>)->Vec<CMatrix2>{
        let mut result = Vec::new();
        for i in input{
            result.push(normalize(&i));
        }
        result
    }

    #[test]
    fn hermitiy_test(){
        let h_matrix = make_random_hermitian(3, 3);
        assert!(check_hermitian(&h_matrix));
    }

    #[test]
    fn make_new_base_test(){
        let ham_a = make_random_hermitian(2, 2);
        let ham_b = make_random_hermitian(2, 2);
        let vec_b = vec![ham_b.clone()];
        let newbase = gs_system(ham_a.clone(), &vec_b);

        assert!(check_linear_ind_intest(&vec_b[0], &newbase),"ham_a {:}, \nham_b {:}\nnewbse {:}",ham_a,ham_b, newbase);
    }

    fn is_vertical(c1:&CMatrix2,c2:&CMatrix2)->bool{
        let result = c1.dot(&c2);
        result.re().pow(2) + result.im().pow(2) < 1.0e-9 * (c1.ncols() * c2.nrows() * 2) as f64
    }

    pub fn is_system_vertical(target: &CMatrix2, system: &Vec<CMatrix2>)->bool{
        for system_mat in system{
            if ! is_vertical(target, system_mat){
                println!("ip val: {:e}",target.dot(&system_mat).abs());
                // println!(
                    // "inner product val : {}\ncom {}not vertical element of system{}",
                //     target.dot(&system_mat).abs(),
                //     target,
                //     system_mat
                //     );
                return false;
            }
        }
        true
    }

    pub fn is_all_vertical(system: &Vec<CMatrix2>)->bool{
        for v_mat in system.iter().combinations(2){
            if ! is_vertical(v_mat[0], v_mat[1]){
                return false;
            }
        }
        true
    }

    pub fn reshape(target: &CMatrix2, row:i16, col:i16)->CMatrix2{
        if (target.ncols() * target.nrows()) != (row * col) as usize{
            assert!(false, "input row*col {}, target {}",row*col,target);
        }

        let mut reshaped_vec = Vec::new();
        for ptr in target.iter(){
            reshaped_vec.push(ptr.clone());
        }
        if reshaped_vec.len() != (row * col) as usize{
            assert_eq!(reshaped_vec.len(), (row*col) as usize, "reshaped_vec {}, row*col {}, target {}",reshaped_vec.len(),row*col,target);
        }
        DMatrix::from_vec(row as usize, col as usize,reshaped_vec)
    }

    fn make_identity(u:usize)->CMatrix2{
        let mut mat = CMatrix2::zeros(u, u);
        for i in 0..u{
            mat[(i,i)] = Complex::new(1.0,0.0);
        }
        mat
    }

    #[test]
    fn check_ajoint_rep_imaginary(){
        let ham_a = make_random_hermitian(4,4);
        let cla = ham_a.clone();
        let ham_b = make_identity(4);
        let clb = ham_b.clone();
        let vector = vec![ham_a,ham_b];
        let dla = super::super::dla::get_dla(&vector);
        if dla.len() > 16{
            assert!(
                false,
                "Too big dla\nsize:{},\nham_a{}ham_b{},",
                dla.len(),
                cla,
                clb,
            )
        }
        let len = dla.len()as i16;
        let basis = get_schmit_basis(dla);
        let adjoint_a = adjoint_rep(vector[0].clone(), &basis);
        println!("😊adjoint_a:\n {:}",reshape(&adjoint_a, len * len , 1));
        assert!(adjoint_a.iter().all(|&z| z.re <= 1.0e-7));
    }

}
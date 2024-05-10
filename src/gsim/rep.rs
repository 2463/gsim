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
        let new_base = make_new_base(element, &sch_basis);
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

// c_n - \sum_i (<c_i,c_n>/<c_i,c_i>)c_i を計算する場所
pub(super) fn make_new_base(target: CMatrix2,basis: &Vec<CMatrix2>)->CMatrix2{
    let mut minus: CMatrix2 = DMatrix::zeros(target.ncols(),target.nrows());

    for base in basis{
        let coef = base.dot(&target) / base.dot(&base);
        minus = (base * coef) + minus;
    }
    target - minus
}

#[cfg(test)]
pub mod test_rep{
    use super::super::dla::test_dla::make_random_hermitian;
    use num::Complex;
    use nalgebra::ComplexField;

    use super::*;

    pub fn check_linear_ind_intest(matrix_a : &CMatrix2, matrix_b : &CMatrix2)->bool{
        (matrix_a.dot(matrix_a) * matrix_b.dot(matrix_b) - matrix_a.dot(matrix_b).powi(2)).abs() > 1.0e-7
    }


    fn make_minus(target:&CMatrix2,base:&CMatrix2)->CMatrix2{
        let mut minus: CMatrix2 = DMatrix::zeros(target.ncols(),target.nrows());
    
        let coef = base.dot(&target) / base.dot(&base);
        minus = (base * coef) + minus;
    
        minus
    }    

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

    #[test]
    fn hermitiy_test(){
        let h_matrix = make_random_hermitian(3, 3);
        assert!(check_hermitian(&h_matrix));
    }

    #[test]
    fn make_new_base_test(){
        let ham_a = make_random_hermitian(3, 3);
        let ham_b = make_random_hermitian(3, 3);
        let vec_b = vec![ham_b.clone()];
        let newbase = make_new_base(ham_a.clone(), &vec_b);

        assert!(check_linear_ind_intest(&vec_b[0], &newbase),"ham_a {:}, \nham_b {:}\nnewbse {:}",ham_a,ham_b, newbase);
    }

    pub fn reshape(target: CMatrix2, row:i16, col:i16)->CMatrix2{
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

    #[test]
    fn make_new_base_test2(){
        let avec = vec![
            Complex::new(0.4,0.0),
            Complex::new(-0.9,0.4),
            Complex::new(-0.2,1.0),
            Complex::new(-0.9,-0.4),
            Complex::new(-0.6,0.0),
            Complex::new(0.3,0.2),
            Complex::new(-0.2,-1.0),
            Complex::new(0.3,-0.2),
            Complex::new(0.5,0.0)];
        let bvec = vec![
                Complex::new(-0.9,0.0),
                Complex::new(0.9,0.9),
                Complex::new(-0.3,1.0),
                Complex::new(0.9,-0.9),
                Complex::new(0.5,0.0),
                Complex::new(0.7,0.3),
                Complex::new(-0.3,-1.0),
                Complex::new(0.7,-0.3),
                Complex::new(0.5,0.0)];
        let a = DMatrix::from_vec(3,3,avec);
        let cl_a = a.clone();
        let re_a = reshape(a.clone(), 9, 1);
        let b = DMatrix::from_vec(3,3,bvec);
        let cl_b = b.clone();
        let re_b = reshape(b.clone(), 9, 1);
        let resultvec = vec![
            Complex::new(-13.03793103,0.0),
            Complex::new(12.53793103,13.83793103),
            Complex::new(-4.67931034,15.93103448),
            Complex::new(12.53793103,-13.83793103),
            Complex::new(6.86551724, 0.0),
            Complex::new(10.75172414,4.67931034),
            Complex::new(-4.67931034,-15.93103448),
            Complex::new(10.75172414,-4.67931034),
            Complex::new(7.96551724,0.0)];
        let minus = make_minus(&a, &b);
        let re_minus = reshape(minus, 9, 1);

        let result = DMatrix::from_vec(9, 1, resultvec);
        let calc_result = make_new_base(a, &vec![b]);
        let reshaped_result = reshape(calc_result, 9, 1);
        assert!(
            (result.clone() - reshaped_result.clone()).norm() < 1.0e-5,
            "result ({},{}) {:}, calc_result ({},{}) {:}, a {:}, b {:}, <a,b> {:}, <b,b> {:}, <a,b>/<b,b> {:},\n <a,b>/<b,b>b {:},a-<a,b>/<b,b>b {:},cl a-<a,b>/<b,b>b {:}, minus {:}, distane {}",
            result.ncols(),
            result.nrows(),
            result,reshaped_result.ncols(),
            reshaped_result.nrows(),
            reshaped_result,
            re_a,
            re_b,
            re_a.dot(&re_b),
            re_b.dot(&re_b),
            re_b.dot(&re_a)/ re_b.dot(&re_b),
            re_b.clone() * (re_b.dot(&re_a) / re_b.dot(&re_b)),
            re_a.clone() - (re_b.clone() * (re_b.dot(&re_a) / re_b.dot(&re_b)) + CMatrix2::zeros(9,1)),
            reshape(cl_a.clone() - (cl_b.clone() * (cl_b.dot(&cl_a) / cl_b.dot(&cl_b)) + CMatrix2::zeros(3,3)), 9, 1),
            re_minus,
            (result.clone() - reshaped_result.clone()));

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
        println!("😊adjoint_a:\n {:}",reshape(adjoint_a.clone(), len * len , 1));
        assert!(adjoint_a.iter().all(|&z| z.re <= 1.0e-7));
    }

}
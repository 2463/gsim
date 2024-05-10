use num::complex::Complex64;
use itertools;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::{ComplexField, DMatrix};

type CMatrix2 = DMatrix<Complex64>;

pub fn generate_dla(vector_of_hamiltonians : &Vec<CMatrix2>)-> Vec<CMatrix2>{
    // プログレスバー作成
    let pb_spinner = ProgressBar::new_spinner();
    let spinner_style = ProgressStyle::with_template("{prefix:.bold.dim} {spinner:.green} {wide_msg}").unwrap();
    pb_spinner.set_style(spinner_style);

    let mut v_o_hs : Vec<CMatrix2> = Vec::new();
    for hamiltonian in vector_of_hamiltonians{
        // v_o_hs の内容がすべて線形独立であるようにする．
        if check_linear_ind_systems(&hamiltonian, &v_o_hs){
            v_o_hs.push(hamiltonian.clone());
        }
    }

    //[Algorithm]
    //1. v_o_hsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
    //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
    //3. new group を old group に加える．new group を空にする（append）
    //4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
    //5. v_o_hs から一つ取り出して new group に入れて 2に戻る v_o_hs が空になったら終了処理(最後のnew で検査する)

    //1. v_o_hsの前から1つ取り出して old group に入れる．
    let mut old = match v_o_hs.pop() {
        None=>return v_o_hs,
        Some(i)=> vec![i],
    };
    let mut new:Vec<CMatrix2> = Vec::new();

    while !v_o_hs.is_empty() {
        // プログレスバーを一つ進める
        // pb.inc(1);

        // (1.2 or 5.) v_o_hs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
        // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
        new.push(match v_o_hs.pop() {
            None=>{return old;},
            Some(i)=>i,
        });
        while !new.is_empty() {
            pb_spinner.set_message(format!("current dim(DLA) : {}",old.len()));
            //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
            let coms = new_old_commutators(&new, &old);
            //3. new group を old group に加える．new group を空にする（append）
            old.append(&mut new);
            // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
            for com in coms{
                pb_spinner.inc(1);
                if is_zero(&com){continue;}
                if check_linear_ind_systems(&com, &old) 
                    & check_linear_ind_systems(&com, &new){new.push(com);}
            }
        }
    }
    old
}

fn is_zero(matrix: &CMatrix2)->bool{
    matrix.norm() < 1.0e-5
}

// アルゴリズムの2.
fn new_old_commutators(new: &Vec<CMatrix2>, old: &Vec<CMatrix2>)->Vec<CMatrix2>{
    let mut com_ham_vec_p: Vec<CMatrix2> = Vec::new();
    for (i,j) in itertools::iproduct!(0..new.len(),0..old.len()){
        com_ham_vec_p.push(commutator(&old[j], &new[i]));
    }
    com_ham_vec_p
}

pub(super) fn commutator(hamiltonian_a :  &CMatrix2, hamiltonian_b : &CMatrix2)-> CMatrix2{
    let com_ham = (hamiltonian_a * hamiltonian_b) - (hamiltonian_b * hamiltonian_a);
    return com_ham;
}

fn check_linear_ind_systems(target : &CMatrix2, system : &Vec<CMatrix2>)->bool{
    for system_matrix in system {
        if !check_linear_ind(target, system_matrix){return false;}
    }
    true
}

fn check_linear_ind(matrix_a : &CMatrix2, matrix_b : &CMatrix2)->bool{
    (matrix_a.dot(matrix_a) * matrix_b.dot(matrix_b) - matrix_a.dot(matrix_b).powi(2)).abs() > 1.0e-7
}

#[cfg(test)]
pub mod test_dla{
    use rand::Rng;
    use num::Complex;
    use super::*;

    macro_rules! round {
        ($x:expr, $scale:expr) => (($x * $scale).round() / $scale)
    }

    pub fn is_vertical(matrix_a:&CMatrix2, matrix_b:&CMatrix2)->bool{
        (matrix_a.dot(&matrix_b)).abs() < 1.0e-7
    }

    pub fn make_random_hermitian(rows:usize,cols:usize)->CMatrix2{
        // ランダムな複素数を含む行列を生成
        let mut rng = rand::thread_rng();
        let mut matrix: CMatrix2 = DMatrix::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                match i.cmp(&j) {
                    std::cmp::Ordering::Less=>{continue;}
                    std::cmp::Ordering::Greater=>{
                        let real_part: f64 = rng.gen_range(-1.0..1.0);
                        let rounded_real = round!(real_part, 10.0);
                        let imag_part: f64 = rng.gen_range(-1.0..1.0);
                        let rounded_imag = round!(imag_part, 10.0);
                        matrix[(i, j)] = Complex64::new(rounded_real, rounded_imag);
                        matrix[(j, i)] = Complex64::new(rounded_real, rounded_imag).conj();
                    }
                    std::cmp::Ordering::Equal=>{
                        let real_part:f64 = rng.gen_range(-1.0..1.0);
                        let rounded_real = round!(real_part, 10.0);
                        matrix[(i,i)] = Complex64::new(rounded_real,0.0);
                    }
                }
            }
        }
        matrix
    }

    fn make_simple_matrix1(rows:usize,cols: usize)->CMatrix2{
        let mut matrix = DMatrix::zeros(rows,cols);
        matrix[(0,0)] = Complex::new(1.0, 0.0);
        matrix[(1,0)] = Complex::new(1.0, 0.0);
        matrix[(1,1)] = Complex::new(1.0, 0.0);
        matrix   
    }

    fn make_simple_matrix2(rows:usize,cols: usize)->CMatrix2{
        let mut matrix = DMatrix::zeros(rows, cols);
        matrix[(0,2)] = Complex::new(0.0, 0.1);
        matrix[(2,0)] = Complex::new(0.0, 0.1);
        matrix[(2,2)] = Complex::new(0.0, 0.1);
        matrix   
    }

    pub fn check_linear_ind_intest(matrix_a : &CMatrix2, matrix_b : &CMatrix2)->bool{
        (matrix_a.dot(matrix_a) * matrix_b.dot(matrix_b) - matrix_a.dot(matrix_b).powi(2)).abs() > 1.0e-7
    }

    pub fn generate_dla_intest(vector_of_hamiltonians : &Vec<CMatrix2>)->Vec<CMatrix2>{
        let pb_spinner = ProgressBar::new_spinner();
        let spinner_style = ProgressStyle::with_template("{prefix:.bold.dim} {spinner:.green} {wide_msg}").unwrap();
        pb_spinner.set_style(spinner_style);
    
        let mut v_o_hs : Vec<CMatrix2> = Vec::new();
        for hamiltonian in vector_of_hamiltonians{
            // v_o_hs の内容がすべて線形独立であるようにする．
            if check_linear_ind_systems(&hamiltonian, &v_o_hs){
                v_o_hs.push(hamiltonian.clone());
            }
        }
    
        //[Algorithm]
        //1. v_o_hsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
        //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
        //3. new group を old group に加える．new group を空にする（append）
        //4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
        //5. v_o_hs から一つ取り出して new group に入れて 2に戻る v_o_hs が空になったら終了処理(最後のnew で検査する)
    
        //1. v_o_hsの前から1つ取り出して old group に入れる．
        let mut old = match v_o_hs.pop() {
            None=>return v_o_hs,
            Some(i)=> vec![i],
        };
        let mut new:Vec<CMatrix2> = Vec::new();
    
        while !v_o_hs.is_empty() {
            // (1.2 or 5.) v_o_hs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
            // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
            new.push(match v_o_hs.pop() {
                None=>{return old;},
                Some(i)=>i,
            });
            while !new.is_empty() {
                pb_spinner.set_message(format!("current dim(DLA) : {}",old.len()));
                // test 用コード
                if old.len() > 16 {return old;}
                //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
                let coms = new_old_commutators(&new, &old);
                //3. new group を old group に加える．new group を空にする（append）
                old.append(&mut new);
                // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
                for com in coms{
                    pb_spinner.inc(1);
                    if is_zero(&com){continue;}
                    if check_linear_ind_systems(&com, &old) 
                        & check_linear_ind_systems(&com, &new){new.push(com);}
                }
            }
        }
        old    
    }

    #[test]
    fn generate_dla_test1(){
        let ham_1 = make_simple_matrix1(3,3);
        let ham_2 = make_simple_matrix1(3,3);
        let vector_of_hamiltonians  = vec![ham_1,ham_2];
        let dla = generate_dla(&vector_of_hamiltonians);
        assert_eq!(dla.len(),1);
    }

    // #[test]
    // fn generate_dla_test_random(){
    //     let ham1 = make_random_hermitian(4, 4);
    //     let clham1 = ham1.clone();
    //     let ham2 = make_random_hermitian(4, 4);
    //     let clham2 = ham2.clone();
    //     let ham3 = make_random_hermitian(4, 4);
    //     let clham3 = ham3.clone();
    //     let dla = generate_dla_intest(&vec![ham1,ham2,ham3]);
    //     if dla.len() > 16 {
    //         assert!(false,
    //             "ham1 {} ham2 {} ham3 {}",
    //             clham1,
    //             clham2,
    //             clham3);
    //     }else if dla.len() < 3 {
    //         assert!(false,
    //             "dla.len() {},ham1 {} ham2 {} ham3 {}",
    //             dla.len(),
    //             clham1,
    //             clham2,
    //             clham3);
    //     }
    // }

    #[test]
    fn generate_dla_test_fixed(){
        let avec = vec![
            Complex::new(0.9,0.0),
            Complex::new(0.0,0.4),
            Complex::new(-0.9,-0.9),
            Complex::new(0.7,1.0),
            Complex::new(-0.0,-0.4),
            Complex::new(-0.7,0.0),
            Complex::new(-0.8, 0.3),
            Complex::new(0.0,-0.5),
            Complex::new(-0.9,0.9),
            Complex::new(-0.8,-0.3),
            Complex::new(-0.8,0.0),
            Complex::new(0.1,-0.5),
            Complex::new(0.7,-1.0),
            Complex::new(0.0,0.5),
            Complex::new(0.1,0.5),
            Complex::new(-0.1,0.0)
            ];
        let bvec = vec![
            Complex::new(0.9,0.0),
            Complex::new(0.5,0.1),
            Complex::new(0.2,-0.1),
            Complex::new(1.0,0.8),
            Complex::new(0.5,-0.1),
            Complex::new(0.9,0.0),
            Complex::new(0.2,0.4),
            Complex::new(1.0,0.3),
            Complex::new(0.2,0.1),
            Complex::new(0.2,-0.4),
            Complex::new(0.1,0.0),
            Complex::new(-0.3,-0.5),
            Complex::new(1.0,-0.8),
            Complex::new(1.0,-0.3),
            Complex::new(-0.3,0.5),
            Complex::new(0.2,0.0)];
        let a = DMatrix::from_vec(4,4,avec);
        let cl_a = a.clone();
        let b = DMatrix::from_vec(4,4,bvec);
        let cl_b = b.clone();
        let dla = generate_dla_intest(&vec![a,b]);
        if dla.len() > 16 {
            assert!(false,
                "dla.len() {},\nham1 {} ham2 {}",
                dla.len(),
                cl_a,
                cl_b);
        }else if dla.len() < 3 {
            assert!(false,
                "dla.len() {},\nham1:{}ham2:{}com12:{}\nham1 com12 ind? {}\nham2 com12 ind? {}",
                dla.len(),
                cl_a,
                cl_b,
                cutoff(commutator(&cl_a,&cl_b)),
                check_linear_ind_intest(&cl_a, &commutator(&cl_a,&cl_b)),
                check_linear_ind_intest(&cl_b, &commutator(&cl_a,&cl_b)),
            );
        }

    }

    // #[test]
    // fn multiple_gdla_test(){
    //     for _ in 0..1000{
    //         let ham1 = make_random_hermitian(4, 4);
    //         let clham1 = ham1.clone();
    //         let ham2 = make_random_hermitian(4, 4);
    //         let clham2 = ham2.clone();
    //         let dla = generate_dla_intest(&vec![ham1,ham2]);
    //         if dla.len() > 16 {
    //             assert!(false,
    //                 "ham1 {} ham2 {}",
    //                 clham1,
    //                 clham2);
    //         }else if dla.len() < 3 {
    //             assert!(false,
    //                 "dla.len() {},\nham1:{}ham2:{}com12:{}com21:{}\nham1 com12 ind? {}\nham2 com12 ind? {}\nham1 com21 ind? {}\nham2 com21 ind? {}",
    //                 dla.len(),
    //                 clham1,
    //                 clham2,
    //                 commutator(&clham1,&clham2),
    //                 commutator(&clham2,&clham1),
    //                 check_linear_ind_intest(&clham1, &commutator(&clham1,&clham2)),
    //                 check_linear_ind_intest(&clham2, &commutator(&clham1,&clham2)),
    //                 check_linear_ind_intest(&clham1, &commutator(&clham2,&clham1)),
    //                 check_linear_ind_intest(&clham2, &commutator(&clham2,&clham1)),
    //             );
    //         }
    //     }
    // }

    fn cutoff(matrix: CMatrix2)->CMatrix2{
        matrix.map(|x|->Complex<f64>{return round!(x,1000.0)})
    }

    #[test]
    fn new_old_commutator_test1(){
        let ham_1 = make_random_hermitian(3,3);
        let ham_2 = make_random_hermitian(3,3);
        let com = new_old_commutators(&vec![ham_1], &vec![ham_2]);
        assert!(com.len() == 1,"new_old_commutator \n{:?}",com);
    }
    #[test]
    fn commutator_test1(){
        let ham_a = make_random_hermitian(3,3);
        let com = commutator(&ham_a, &ham_a);
        // 勝手に move してないかチェック
        // println!("{}",ham_a);
        assert!(com.norm_squared() < 1.0e-7, "commutator of same matrices should be 0 but its value is {:}\n ham_a is {:}",com, ham_a)
    }

    #[test]
    fn commutator_test2(){
        let ham_a = make_random_hermitian(3,3);
        let ham_b = make_random_hermitian(3,3);
        let com = commutator(&ham_a, &ham_b);
        
        fn comcalc(ham_a: &CMatrix2, ham_b: &CMatrix2)->CMatrix2{
            let col = ham_a.ncols();
            let row = ham_a.nrows();
            let mut result = CMatrix2::zeros(row,col);
            for (i,j) in itertools::iproduct!(0..row,0..col){
                result[(i,j)] = (ham_a.row(i) * ham_b.column(j) - ham_b.row(i) * ham_a.column(j))[(0,0)]
            }
            result
        }

        assert_eq!(com,comcalc(&ham_a, &ham_b),"com {:}ham_a {:}ham_b {:}",com,ham_a,ham_b);

    }

    #[test]
    fn check_linear_ind_systems_test1(){
        let e: CMatrix2 = make_simple_matrix1(3, 3);
        let system : Vec<CMatrix2> = vec![e;3];
        let target = make_simple_matrix2(3,3);
        assert_eq!(true,check_linear_ind_systems(&target, &system))
    }

    #[test]
    fn check_linear_ind_systems_test2(){
        let e = make_simple_matrix1(3,3);
        let system : Vec<CMatrix2> = vec![e;3];
        let target = make_simple_matrix2(3,3);
        assert_eq!(true,check_linear_ind_systems(&target, &system))
    }
    #[test]
    fn check_linear_ind_test(){
        let a : CMatrix2 = make_simple_matrix1(3,3);
        let b = make_simple_matrix2(3,3);
        assert_eq!(true, check_linear_ind(&a, &b));
    }
    // #[test]
    // fn linear_ind_random_test(){
    //     let a = make_random_hermitian(3,3);
    //     let b = make_random_hermitian(3,3);
    //     assert_eq!(true, check_linear_ind_part(&a, &b),"a : {:}\n b : {:}",a,b);
    // }

}
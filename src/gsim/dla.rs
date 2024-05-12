/// # dla_gs.rs
/// dla を求める途中で Gram-Schmidt で直交化させていく関数
use num::complex::Complex64;
use nalgebra::DMatrix;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use super::rep;
use itertools::Itertools;


type CMatrix2 = DMatrix<Complex64>;

pub fn get_dla(vector_of_hamiltonians: &Vec<CMatrix2>)->Vec<CMatrix2>{
    // プログレスバー作成
    let m = MultiProgress::new();
    let pb_spinner = m.add(ProgressBar::new_spinner());
    let spinner_style = ProgressStyle::with_template("{prefix:.bold.dim} {spinner:.green} {wide_msg}").unwrap();
    pb_spinner.set_style(spinner_style);


    // まず入力を gs で直交化させておく．ここでハミルトニアンは後で使うので clone する
    let mut gs_vohs = rep::get_schmit_basis(vector_of_hamiltonians.clone());
    // 直交化した gs で以下のアルゴリズムを実行
    //[Algorithm]
    //1. gs_vohsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
    //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
    //3. new group を old group に加える．new group を空にする（append）
    //4. commutator が 0 でなく，old group と new group に対して独立だったら（ここで gs を使用） new groupに加える．これをcomm group すべてに実行
    //5. gs_vohs から一つ取り出して new group に入れて 2に戻る gs_vohs が空になったら終了処理(最後のnew で検査する)

    // 1 の前半を実行する部分
    let mut old = match gs_vohs.pop() {
        None=> return gs_vohs,
        Some(i)=> vec![i]
    };
    let mut new = Vec::new();

    while !gs_vohs.is_empty() {
        // (1.2 or 5.) gs_vohs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
        // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
        new.push(match gs_vohs.pop() {
            None=>{return old;},
            Some(i)=>i
        });
        while !new.is_empty() {
            //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
            let coms = [new_old_commutators(&new, &old),new_new_commutator(&new)].concat();
            //3. new group を old group に加える．new group を空にする（append）
            old.append(&mut new);

            // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
            for com in coms{
                pb_spinner.set_message(format!("current dim(DLA) : {}, candidate num : {}",old.len(),new.len()));
                pb_spinner.inc(1);
                if is_zero(&com){println!("is not ind");continue;}
                let gs_com = rep::make_new_base(com,&old);
                let gs_com = rep::make_new_base(gs_com,&new);
                //  gram schmidt の結果が 0 なら線形従属である
                if is_zero(&(gs_com)){println!("is not ind from zero");continue;}
                new.push(gs_com);
            }
        }
    }
    old
}

pub fn is_zero(matrix: &CMatrix2)->bool{
    matrix.norm() < 1.0e-9
}

fn new_old_commutators(new: &Vec<CMatrix2>, old: &Vec<CMatrix2>)->Vec<CMatrix2>{
    let mut com_ham_vec_p: Vec<CMatrix2> = Vec::new();
    for (i,j) in itertools::iproduct!(0..new.len(),0..old.len()){
        com_ham_vec_p.push(commutator(&old[j], &new[i]));
    }
    com_ham_vec_p
}

fn new_new_commutator(new: &Vec<CMatrix2>)->Vec<CMatrix2>{
    let mut com_vec = Vec::new();
    for hams in new.iter().combinations(2){
        com_vec.push(commutator(hams[0],hams[1]));
    }
    com_vec
}

pub(super) fn commutator(hamiltonian_a :  &CMatrix2, hamiltonian_b : &CMatrix2)-> CMatrix2{
    let com_ham = (hamiltonian_a * hamiltonian_b) - (hamiltonian_b * hamiltonian_a);
    return com_ham;
}

#[cfg(test)]
pub mod test_dla{

    use super::*;
    use rand::Rng;
    use num::Complex;
    use nalgebra::ComplexField;
    use colored::*;

    macro_rules! round {
        ($x:expr, $scale:expr) => (($x * $scale).round() / $scale)
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

    fn check_linear_ind(matrix_a : &CMatrix2, matrix_b : &CMatrix2)->bool{
        (matrix_a.dot(matrix_a) * matrix_b.dot(matrix_b) - matrix_a.dot(matrix_b).powi(2)).abs() > 1.0e-7
    }

    fn check_linear_ind_systems(target : &CMatrix2, system : &Vec<CMatrix2>)->bool{
        for system_matrix in system {
            if !check_linear_ind(target, system_matrix){
                // println!("{} and {} is not independent",cutoff(&target),cutoff(&system_matrix));
                return false;
            }
        }
        true
    }
    #[test]
    fn test_get_dla(){
        let n = 6;
        let hdim = 2i64.pow(n) as usize;
        let dladim = 4i64.pow(n) as usize;
        let rand1 = make_random_hermitian(hdim,hdim);
        let rand2 = make_random_hermitian(hdim,hdim);
        let vohs = vec![rand1,rand2];
        let dla = get_dla(&vohs);
        println!("vohs");
        print_all(&vohs);
        println!("bs_vohs");
        print_all(&rep::get_schmit_basis(vohs));
        println!("dla.len() : {}",dla.len());
        println!("dla");
        print_all(&dla);
        assert!(dla.len() < dladim+1);
    }

    #[test]
    fn test_get_dla_fixed(){
        let avec = vec![
            Complex::new(-0.5,0.0),
            Complex::new(-0.9,-0.2),
            Complex::new(-0.9,0.2),
            Complex::new(0.7,0.0)];
            // [-0.5+0j, -0.9-0.2j,-0.9+0.2j, 0.7+0j]
        let bvec = vec![
                Complex::new(0.6,0.0),
                Complex::new(-0.5,-0.3),
                Complex::new(-0.5,0.3),
                Complex::new(-0.4,0.0)];
            // [0.6+0j, -0.5-0.3j,-0.5+0.3j,   -0.4+0j]
        let a = DMatrix::from_vec(2,2,avec).conjugate();
        let b = DMatrix::from_vec(2,2,bvec).conjugate();
        let vohs = vec![a,b];
        let dla = get_dla(&vohs);
        println!("vohs");
        print_all(&vohs);
        println!("bs_vohs");
        print_all(&rep::get_schmit_basis(vohs));
        println!("dla.len() : {}",dla.len());
        println!("dla");
        print_all(&dla);
        assert!(dla.len() < 5);
    }

    fn print_all(vec_cmat:&Vec<CMatrix2>){
        for cmat in vec_cmat{
            let simplified = cmat;
            print_cmatrix_in_python_form(&simplified);
        }
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

    fn print_cmatrix_in_python_form(mat: &CMatrix2){
        print!("[");
        for elem in mat.iter(){
            print_complex_in_python_form(elem);
            print!(",")
        }
        print!("]\n");
    }

    fn print_complex_in_python_form(c: &Complex64){
        print!("{}+{}j",c.re,c.im)
    }

    fn cutoff(matrix: &CMatrix2)->CMatrix2{
        matrix.map(|x|->Complex<f64>{return round!(x,1000.0)})
    }

    pub fn get_dla_intest(vector_of_hamiltonians: &Vec<CMatrix2>)->Vec<CMatrix2>{
        // プログレスバー作成
        let pb_spinner = ProgressBar::new_spinner();
        let spinner_style = ProgressStyle::with_template("{prefix:.bold.dim} {spinner:.green} {wide_msg}").unwrap();
        pb_spinner.set_style(spinner_style);
    
    
        // まず入力を gs で直交化させておく．ここでハミルトニアンは後で使うので clone する
        let mut gs_vohs = rep::get_schmit_basis(vector_of_hamiltonians.clone());
        // 直交化した gs で以下のアルゴリズムを実行
        //[Algorithm]
        //1. gs_vohsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
        //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
        //3. new group を old group に加える．new group を空にする（append）
        //4. commutator が 0 でなく，old group と new group に対して独立だったら（ここで gs を使用） new groupに加える．これをcomm group すべてに実行
        //5. gs_vohs から一つ取り出して new group に入れて 2に戻る gs_vohs が空になったら終了処理(最後のnew で検査する)
    
        // 1 の前半を実行する部分
        let mut old = match gs_vohs.pop() {
            None=> return gs_vohs,
            Some(i)=> vec![i]
        };
        let mut new = Vec::new();
    
        while !gs_vohs.is_empty() {
            // (1.2 or 5.) gs_vohs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
            // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
            new.push(match gs_vohs.pop() {
                None=>{return old;},
                Some(i)=>i
            });
            let mut iteration_num = 0;
            while !new.is_empty() {
                println!("###### {}-th iteration ########",iteration_num);
                iteration_num += 1;
                //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
                let coms = [new_old_commutators(&new, &old),new_new_commutator(&new)].concat();
                //3. new group を old group に加える．new group を空にする（append）
                old.append(&mut new);
                if old.len() > 17 {return old;}

                // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
                for com in coms{
                    println!("com:{}",generate_cmatrix_string_in_python_form(&com).blue());
                    pb_spinner.set_message(format!("current dim(DLA) : {}, candidate num : {}",old.len(),new.len()));
                    pb_spinner.inc(1);
                    if is_zero(&com){println!("is not ind");continue;}
                    let gs_com = rep::make_new_base(com,&old);
                    let gs_com = rep::make_new_base(gs_com,&new);
                    //  gram schmidt の結果が 0 なら線形従属である
                    println!("gs_com:{}",generate_cmatrix_string_in_python_form(&gs_com).green());
                    println!(
                        "is ind? {}, full.len {}",
                        format!("{}",check_linear_ind_systems(&gs_com, &old) & check_linear_ind_systems(&gs_com, &new)).red(),
                        old.len() + new.len());
                    // if !(check_linear_ind_systems(&gs_com, &old) & check_linear_ind_systems(&gs_com, &new)) {
                    //     // println!("print full");
                    //     // print_all(&full);
                    //     // println!("target {}, before com {}",cutoff(&gs_com), cutoff(&com_cl));
                    //     continue;
                    // }
                    if is_zero(&(gs_com)){println!("is not ind from zero");continue;}
                    new.push(gs_com);
                }
            }
        }
        old
    }
}

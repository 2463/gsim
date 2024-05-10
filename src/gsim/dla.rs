/// # dla_gs.rs
/// dla を求める途中で Gram-Schmidt で直交化させていく関数
use num::complex::Complex64;
use nalgebra::DMatrix;
use indicatif::{ProgressBar, ProgressStyle};
use super::rep;


type CMatrix2 = DMatrix<Complex64>;

pub fn get_dla(vector_of_hamiltonians: &Vec<CMatrix2>)->Vec<CMatrix2>{
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
    let mut full: Vec<CMatrix2> = Vec::new();
    let mut old = match gs_vohs.pop() {
        None=> return gs_vohs,
        Some(i)=> {
            full.push(i.clone());
            vec![i]
        },
    };
    let mut new = Vec::new();

    while !gs_vohs.is_empty() {
        // (1.2 or 5.) gs_vohs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
        // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
        new.push(match gs_vohs.pop() {
            None=>{return old;},
            Some(i)=>{
                full.push(i.clone());
                i},
        });
        while !new.is_empty() {
            pb_spinner.set_message(format!("current dim(DLA) : {}, candidate num : {}",old.len(),new.len()));
            pb_spinner.inc(1);
            //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
            let coms = new_old_commutators(&new, &old);
            //3. new group を old group に加える．new group を空にする（append）
            old.append(&mut new);
            // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
            for com in coms{
                if is_zero(&com){continue;}
                let gs_com = rep::make_new_base(com,&full);
                //  gram schmidt の結果が 0 なら線形従属である
                if is_zero(&(gs_com)){continue;}
                full.push(gs_com.clone());
                new.push(gs_com);
            }
        }
    }
    old
}

fn is_zero(matrix: &CMatrix2)->bool{
    matrix.norm() < 1.0e-5
}

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

#[cfg(test)]
pub mod test_dla{

    use super::*;
    use rand::Rng;

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

    #[test]
    fn test_get_dla_gs(){
        let n = 3;
        let hdim = 2i64.pow(n) as usize;
        let dladim = 4i64.pow(n) as usize;
        let rand1 = make_random_hermitian(hdim,hdim);
        let rand2 = make_random_hermitian(hdim,hdim);
        let vohs = vec![rand1,rand2];
        let dla = get_dla(&vohs);
        assert!(dla.len() > dladim);
        println!("dla.len() : {}",dla.len());
    }
}

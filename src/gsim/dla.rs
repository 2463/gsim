use std::vec;

use super::rep;
use nalgebra::DMatrix;
use num::complex::Complex64;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::Itertools;

type CMatrix2 = DMatrix<Complex64>;

/// # dla_gs.rs
/// dla を求める途中で Gram-Schmidt で直交化させていく関数

pub(super) fn get_dla(vector_of_hamiltonians: &Vec<CMatrix2>)->Vec<CMatrix2>{
        // プログレスバー作成
        let m = MultiProgress::new();

        // ここでハミルトニアンは後で使うので clone する．
        let mut vohs = vector_of_hamiltonians.clone();
        // さらに虚数化する．
        imaginalize(&mut vohs);
        // 以下のアルゴリズムを実行
        //[Algorithm]
        //1. gs_vohsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
        //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
        //3. new group を old group に加える．new group を空にする（append）
        //4. commutator が 0 でなく，old group と new group に対して独立だったら（ここで gs を使用） new groupに加える．これをcomm group すべてに実行
        //5. gs_vohs から一つ取り出して new group に入れて 2に戻る gs_vohs が空になったら終了処理(最後のnew で検査する)
    
        // 1 の前半を実行する部分
        let mut old = match vohs.pop() {
            None=> return vohs,
            Some(i)=> vec![i]
        };
        let mut new = Vec::new();

        
        // 最大のDLA 次元を求めておく
        let maximum = old[0].ncols() * old[0].ncols();
        
        
        while !vohs.is_empty() {
            // println!("old len {}, new len {}",old.len(),new.len());
            // (1.2 or 5.) gs_vohs から一つ取り出して new group に入れて 2に戻る．(最後の new で検査)
            // vohsにもし内容がなかったら，それはvohs に要素が1つしかなかったということなので，oldを返せば良い
            // また，vohs から candidate を取ったとき，それが old と独立になっていなければ，それを加える必要はないので無視する
            new.push(match vohs.pop() {
                None=>{return old;},
                Some(candidate)=>{
                    let gs_candidate = rep::gs_system(candidate,&old);
                    if is_zero(&gs_candidate){continue;}
                    gs_candidate
                    }
            });

            // pb で使う用パラメータ
            let mut iternum = 0;
            while !new.is_empty() {
                // println!("new len {}",new.len());
                // pb
                iternum += 1;
                let pblen = ((new.len() * (new.len() - 1) / 2) + new.len() * old.len()) * 2;
                let pb = m.add(ProgressBar::new(pblen as u64));
                let bar_style = ProgressStyle::with_template(
                    "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos:>7}/{len:7} ({eta})\n{msg}"
                )
                .unwrap();
                pb.set_style(bar_style);
            
                //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
                let mut coms = new_old_commutators(&new, &old, &pb);
                coms.extend(new_new_commutator(&new, &pb));
                // println!("com len {}",coms.len());
                //3. new group を old group に加える．new group を空にする（append）
                // println!("append in old : {}",old.len());
                old.append(&mut new);

                pb.set_message("checking independency of commutators");
            
                // 4. commutator が 0 でなく，old group と new group に対して独立だったら new groupに加える．これをcomm group すべてに実行
                for com in coms{
                    pb.set_message(format!("checking commutator independency : current dim(DLA) : {}, candidate num : {}, current iteration : {}",old.len(),new.len(),iternum));
                    pb.inc(1);

                    if is_zero(&com){continue;}
                    let gs_com = rep::gs_system(com,&old);
                    let gs_com = rep::gs_system(gs_com,&new);
                    // println!("is_zero? : {}",gs_com.norm_squared());
                    //  gram schmidt の結果が 0 なら線形従属である
                    if is_zero(&(gs_com)){continue;}
                    if old.len() + new.len() == maximum as usize{
                        old.append(&mut new);
                        return old;
                    }
                    let gs_com = rep::smallize(&gs_com);
                    new.push(gs_com);
                    // println!("pushed in new : {}",new.len());
                }
                pb.finish_and_clear();
                // println!("# all commutators are checked");
            }
        }
        // new が大きすぎないか検査しておく
        let error = format!("Too Large DLA Error | number of dla : {} (Independency error occurred.)",old.len());
        if (old.len()) > maximum as usize{panic!("{}",error)}
        rep::normalize_all(old)
}

pub(super) fn is_zero(matrix: &CMatrix2)->bool{
    matrix.norm_squared() < 1.0e-7 * (matrix.ncols() * matrix.nrows() * 2) as f64
}

fn imaginalize(vector: &mut Vec<CMatrix2>){
    for i in 0..vector.len(){
        vector[i] = &vector[i] * Complex64::new(0.0, 1.0);
    }
}

fn new_old_commutators(new: &Vec<CMatrix2>, old: &Vec<CMatrix2>, pb: &ProgressBar)->Vec<CMatrix2>{
    let mut com_ham_vec_p: Vec<CMatrix2> = Vec::new();
    pb.set_message("generating commutators among new and old generators");

    for (i,j) in itertools::iproduct!(0..new.len(),0..old.len()){
        pb.inc(1);
        com_ham_vec_p.push(commutator(&old[j], &new[i]));
    }
    com_ham_vec_p
}

fn new_new_commutator(new: &Vec<CMatrix2>, pb : &ProgressBar)->Vec<CMatrix2>{
    pb.set_message("generating commutators among new generators");

    let mut com_vec = Vec::new();
    for hams in new.iter().combinations(2){
        pb.inc(1);
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
    use super::rep::test_rep;
    use rand::Rng;

    macro_rules! round {
        ($x:expr, $scale:expr) => (($x * $scale).round() / $scale)
    }

    pub(crate) fn make_random_hermitian(rows:usize,cols:usize)->CMatrix2{
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

    const NUMBER_OF_QUBIT:usize = 1;

    #[test]
    fn test_get_dla(){
        let n = NUMBER_OF_QUBIT;
        let hdim = 2i64.pow(n as u32) as usize;
        let rand1 = make_random_hermitian(hdim,hdim);
        let rand2 = make_random_hermitian(hdim,hdim);
        let vohs = vec![rand1,rand2];
        let dla = get_dla(&vohs);
        // println!("dla.len() : {}",dla.len());
        // println!("dla");
        print_all(&dla);
        let normalized_dla = test_rep::normalize_all(dla);
        assert!(test_rep::is_all_vertical(&normalized_dla));
    }

    pub(crate) fn print_all(vec_cmat:&Vec<CMatrix2>){
        for cmat in vec_cmat{
            let simplified = cmat;
            print_cmatrix_in_python_form(&simplified);
            print!(",");
        }
    }

    pub(crate) fn print_cmatrix_in_python_form(mat: &CMatrix2){
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

}
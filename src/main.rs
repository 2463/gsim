// use gsim::gsim::dla::generate_dla;
// use nalgebra::DMatrix;
// use num::complex::Complex64;
// use rand::Rng;

// type CMatrix2 = DMatrix<Complex64>;

// fn main(){
//     let ham_1 = make_random_hermitian(8,8);
//     let mut vector_of_hamiltonians  = vec![ham_1];
//     vector_of_hamiltonians.push(make_random_hermitian(8, 8));
//     vector_of_hamiltonians.push(make_random_hermitian(8, 8));
//     vector_of_hamiltonians.push(make_random_hermitian(8, 8));
//     let dla = generate_dla(&vector_of_hamiltonians);
//     println!("dla size is {}",dla.len());
// }

// pub fn make_random_hermitian(rows:usize,cols:usize)->CMatrix2{
//     // ランダムな複素数を含む行列を生成
//     let mut rng = rand::thread_rng();
//     let mut matrix: CMatrix2 = DMatrix::zeros(rows, cols);
//     for i in 0..rows {
//         for j in 0..cols {
//             match i.cmp(&j) {
//                 std::cmp::Ordering::Less=>{continue;}
//                 std::cmp::Ordering::Greater=>{
//                     let real_part: f64 = rng.gen_range(-1.0..1.0);
//                     let imag_part: f64 = rng.gen_range(-1.0..1.0);
//                     matrix[(i, j)] = Complex64::new(real_part, imag_part);
//                     matrix[(j, i)] = Complex64::new(real_part, imag_part).conj();        
//                 }
//                 std::cmp::Ordering::Equal=>{
//                     let random = rng.gen_range(-1.0..1.0);
//                     matrix[(i,i)] = Complex64::new(random,0.0);    
//                 }
//             }
//         }
//     }
//     matrix
// }

fn main(){
    println!("test");
}
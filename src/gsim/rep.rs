use crate::gsim::dla::{self, is_zero};
use super::dla::commutator;
use nalgebra::{ComplexField, DMatrix};
use num::complex::Complex64;

type CMatrix2 = DMatrix<Complex64>;

// DLA からシュミット直交基底を得る
// 共役表現でハミルトニアンを低次元に写像する

pub fn get_schmit_basis(mut dla: Vec<CMatrix2>)->Vec<CMatrix2>{
    let first_element = dla.pop().expect("The vector is empty");

    let mut sch_basis: Vec<CMatrix2> = vec![first_element];
    while let Some(element) = dla.pop() {
        let new_base = gs_system(element, &sch_basis);
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
    println!("gs [input]: {}", generate_cmatrix_string_in_python_form(&gs));
        for base in system{
        gs = gram_schmidt(gs, base);
        if dla::is_zero(&gs){
            println!("gs : return zero");
            return CMatrix2::zeros(gs.nrows(), gs.ncols());
        }    
        println!("gs : {}", generate_cmatrix_string_in_python_form(&gs));
    }
    println!("gs [return]: {}",generate_cmatrix_string_in_python_form(&normalize(&gs)));
    normalize(&gs)
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

fn gram_schmidt(target: CMatrix2,base: &CMatrix2)->CMatrix2{
    let result = &target - base * (base.dot(&target) / base.dot(&base));
    result
}

pub fn normalize(c:&CMatrix2)->CMatrix2{
    c * Complex64::new(1.0 / c.norm(),0.0)
}

#[cfg(test)]
pub mod test_rep{
    use super::super::dla::test_dla::*;
    use num::Complex;
    use nalgebra::ComplexField;

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

//     #[test]
//     fn make_new_base_test_fixed1(){
//         let avec = vec![
//             Complex::new(-0.5,0.0),
//             Complex::new(-0.9,-0.2),
//             Complex::new(-0.9,0.2),
//             Complex::new(0.7,0.0)];
//             // [-0.5+0j, -0.9-0.2j,-0.9+0.2j, 0.7+0j]
//         let bvec = vec![
//                 Complex::new(0.6,0.0),
//                 Complex::new(-0.5,-0.3),
//                 Complex::new(-0.5,0.3),
//                 Complex::new(-0.4,0.0)];
//             // [0.6+0j, -0.5-0.3j,-0.5+0.3j,   -0.4+0j]
//         let a = DMatrix::from_vec(2,2,avec).conjugate();
//         let cla = a.clone();
//         let b = DMatrix::from_vec(2,2,bvec).conjugate();
//         let clb = b.clone();
//         let resultvec = vec![
//             Complex::new(-0.42402543,0.0),
//             Complex::new(-0.51511237,-0.08480509),
//             Complex::new(-0.51511237,0.08480509),
//             Complex::new(0.52453516,0.)];
//             // [-0.42402543+0.j        , -0.51511237-0.08480509j,-0.51511237+0.08480509j,  0.52453516+0.j        ]
//         let result = DMatrix::from_vec(2, 2, resultvec).conjugate();
//         let rust_result = make_new_base(a, &vec![b]);
//         let coeff = clb.dot(&cla) / clb.dot(&clb);
//         println!("coeff {}",coeff);
//         println!("coeff_bottom {}",clb.dot(&clb));
//         println!("coeff_over {}",clb.dot(&cla));
//         println!("dotted matrix {:}",(&clb * coeff));
//         let gs = &cla - (&clb * coeff);
//         println!("gs matrix {:}", gs);
//         println!("normalized gs matrix {:}", normalize(&gs));
//         println!("make_new_base {:}",rust_result);
//         println!("target result {:}",result);
//         println!("is vertical? {}",is_vertical(&gs, &clb));
//         assert!(is_close(&rust_result,&result));
//     }

//     #[test]
//     fn make_new_base_test_fixed2(){
// // [-0.424+-0j, -0.515-0.085j, -0.515+0.085j, 0.525+0j]
// // [0.6+-0j, -0.5-0.3j, -0.5+0.3j, -0.4+0j]
// // [0+0.215j, 0.649-0.232j,-0.649-0.232j,0-0.072j]
// // [-0.672+0j, -0.088+0.039j,-0.088-0.039j,-0.728+0j]
// // [0.609+0j, 0.305-0.19j,0.305+0.19j,-0.609+0j]
//         // let target = 
//         let avec = vec![
//             Complex::new(-0.5,0.0),
//             Complex::new(-0.9,-0.2),
//             Complex::new(-0.9,0.2),
//             Complex::new(0.7,0.0)];
//             // [-0.5+0j, -0.9-0.2j,-0.9+0.2j, 0.7+0j]
//         let bvec = vec![
//                 Complex::new(0.6,0.0),
//                 Complex::new(-0.5,-0.3),
//                 Complex::new(-0.5,0.3),
//                 Complex::new(-0.4,0.0)];
//             // [0.6+0j, -0.5-0.3j,-0.5+0.3j,   -0.4+0j]
//         let a = DMatrix::from_vec(2,2,avec).conjugate();
//         let cla = a.clone();
//         let b = DMatrix::from_vec(2,2,bvec).conjugate();
//         let clb = b.clone();
//         let resultvec = vec![
//             Complex::new(-0.42402543,0.0),
//             Complex::new(-0.51511237,-0.08480509),
//             Complex::new(-0.51511237,0.08480509),
//             Complex::new(0.52453516,0.)];
//             // [-0.42402543+0.j        , -0.51511237-0.08480509j,-0.51511237+0.08480509j,  0.52453516+0.j        ]
//         let result = DMatrix::from_vec(2, 2, resultvec).conjugate();
//         let rust_result = make_new_base(a, &vec![b]);
//         let coeff = clb.dot(&cla) / clb.dot(&clb);
//         println!("coeff {}",coeff);
//         println!("coeff_bottom {}",clb.dot(&clb));
//         println!("coeff_over {}",clb.dot(&cla));
//         println!("dotted matrix {:}",(&clb * coeff));
//         let gs = &cla - (&clb * coeff);
//         println!("gs matrix {:}", gs);
//         println!("normalized gs matrix {:}", normalize(&gs));
//         println!("make_new_base {:}",rust_result);
//         println!("target result {:}",result);
//         println!("is vertical? {}",is_vertical(&gs, &clb));
//         assert!(is_close(&rust_result,&result));
//     }

    #[test]
    fn make_new_base_test_fixed3(){
        let mut vector_of_cmat: Vec<CMatrix2> = Vec::new();
        let row = 4;
        let col = 4;
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.11937782179175174,0.00000000000000001295339587568252),
            Complex::new(0.3155934843358782,0.09624670710159047),
            Complex::new(-0.06753221990001108,-0.21934677723428764),
            Complex::new(0.15181455733427657,-0.31001011182446003),
            Complex::new(0.31559348433587814,-0.0962467071015905),
            Complex::new(-0.13320331562954943,0.000000000000000038860187627047566),
            Complex::new(0.017547742178743116,0.20179903505554453),
            Complex::new(-0.13320331562954943,0.24620014026539444),
            Complex::new(-0.06753221990001106,0.21934677723428767),
            Complex::new(0.017547742178743116,-0.20179903505554458),
            Complex::new(-0.22493014974570583,0.0),
            Complex::new(0.22306902557523314,0.010103245496852258),
            Complex::new(0.15181455733427662,0.31001011182446003),
            Complex::new(-0.13320331562954943,-0.24620014026539433),
            Complex::new(0.2230690255752331,-0.010103245496852258),
            Complex::new(-0.13240569098506078,-0.000000000000000017271194500910033),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(0.3,0.0),
            Complex::new(-0.5,-0.2),
            Complex::new(0.4,0.3),
            Complex::new(0.1,0.8),
            Complex::new(-0.5,0.2),
            Complex::new(0.9,0.0),
            Complex::new(-0.4,0.1),
            Complex::new(0.9,-0.2),
            Complex::new(0.4,-0.3),
            Complex::new(-0.4,-0.1),
            Complex::new(0.0,0.0),
            Complex::new(-0.1,-0.8),
            Complex::new(0.1,-0.8),
            Complex::new(0.9,0.2),
            Complex::new(-0.1,0.8),
            Complex::new(-0.4,0.0),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(0.0000000000000000900842476687237,0.1381190576874865),
            Complex::new(0.08936717111918792,-0.31962273316601003),
            Complex::new(-0.22336863643142107,0.058368832553452425),
            Complex::new(-0.2998544105743205,-0.16903696716975528),
            Complex::new(-0.08936717111918836,-0.31962273316601),
            Complex::new(0.00000000000000013098249650549753,0.12146811347291951),
            Complex::new(0.2193204927864357,-0.0026232624831785535),
            Complex::new(0.2200319315102957,0.1456910932236391),
            Complex::new(0.2233686364314212,0.05836883255345213),
            Complex::new(-0.21932049278643576,-0.002623262483178306),
            Complex::new(0.00000000000000014256696968961673,0.1940972990375851),
            Complex::new(0.02356363759313146,-0.2281565513174617),
            Complex::new(0.29985441057432033,-0.16903696716975572),
            Complex::new(-0.22003193151029546,0.1456910932236394),
            Complex::new(-0.023563637593131743,-0.22815655131746165),
            Complex::new(0.00000000000000005856724144361424,0.15463927959499382),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.10403065709650938,-0.00000000000000019665399240308062),
            Complex::new(-0.18474229931482034,0.002191021166602533),
            Complex::new(0.044790304232171174,-0.038211938065054074),
            Complex::new(0.013004467658913946,0.03829062527615257),
            Complex::new(-0.18474229931481914,-0.002191021166599882),
            Complex::new(-0.025980695424620174,-0.00000000000000040918735609841084),
            Complex::new(-0.04339900932976798,-0.06384526204312482),
            Complex::new(-0.3206618896698169,-0.34072430395575753),
            Complex::new(0.04479030423216926,0.038211938065054095),
            Complex::new(-0.043399009329765224,0.06384526204312439),
            Complex::new(0.4676378985678411,-0.0000000000000010858702221688273),
            Complex::new(-0.17411482796254943,0.2979639100798458),
            Complex::new(0.01300446765891159,-0.03829062527615139),
            Complex::new(-0.3206618896698146,0.340724303955757),
            Complex::new(-0.1741148279625501,-0.2979639100798436),
            Complex::new(-0.05826364327553655,-0.0000000000000010217206546464612),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.20792958323234248,0.0000000000000033418998879374634),
            Complex::new(0.12491957000624583,0.005270994291197146),
            Complex::new(0.0007566167647830831,-0.11780257822756601),
            Complex::new(0.010237397338966473,-0.25325903576608894),
            Complex::new(0.12491957000623795,-0.00527099429121594),
            Complex::new(-0.3398892032293166,0.0000000000000028938022295095784),
            Complex::new(-0.04493708968174387,-0.0520880197458461),
            Complex::new(-0.282985009580238,-0.2382992464253663),
            Complex::new(0.0007566167647944036,0.11780257822757105),
            Complex::new(-0.04493708968175613,0.05208801974584573),
            Complex::new(0.2616036370289506,0.000000000000007742583950862852),
            Complex::new(-0.19434695324828946,0.3363855932533712),
            Complex::new(0.010237397338983822,0.2532590357660749),
            Complex::new(-0.28298500958025435,0.23829924642537126),
            Complex::new(-0.1943469532482869,-0.33638559325338513),
            Complex::new(0.015548366527732078,0.0000000000000037324983807259864),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(0.000000000000125345456607092,0.2058769405952003),
            Complex::new(0.005345670889968068,-0.1214355617044896),
            Complex::new(-0.1152362888054063,-0.002706387508106734),
            Complex::new(-0.25308069387810744,-0.009157947663889159),
            Complex::new(-0.005345670890133258,-0.12143556170447518),
            Complex::new(0.00000000000020205924390347943,0.34478136439118545),
            Complex::new(-0.06720449747213135,0.0477157642766525),
            Complex::new(-0.2373774555621052,0.2810253676046789),
            Complex::new(0.11523628880541052,-0.002706387508256343),
            Complex::new(0.06720449747218367,0.04771576427660383),
            Complex::new(-0.0000000000001457431228334263,-0.25463356749995736),
            Complex::new(0.3333002088638142,0.20284461479113247),
            Complex::new(0.253080693878081,-0.009157947664203586),
            Complex::new(0.2373774555624425,0.28102536760441593),
            Complex::new(-0.33330020886360034,0.2028446147915245),
            Complex::new(-0.000000000000005274911136446157,-0.028425259228244876),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.000000000000042612601671943893,0.0919235671495879),
            Complex::new(-0.17354193741649723,-0.20356095834231502),
            Complex::new(0.055078462607971154,-0.011080477364097771),
            Complex::new(-0.34845226262452594,0.09769193828108542),
            Complex::new(0.17354193741646048,-0.20356095834227286),
            Complex::new(-0.00000000000008702790647716897,0.46092190853007114),
            Complex::new(0.031033776393640126,-0.03985035564215031),
            Complex::new(-0.10807451756745988,0.13649643899257505),
            Complex::new(-0.05507846260794051,-0.011080477364093868),
            Complex::new(-0.031033776393671293,-0.039850355642022765),
            Complex::new(0.00000000000011373384560529388,-0.3306301566650737),
            Complex::new(0.29104846145544744,0.0590347653850041),
            Complex::new(0.34845226262445894,0.09769193828113741),
            Complex::new(0.10807451756732107,0.13649643899280448),
            Complex::new(-0.29104846145564855,0.05903476538479615),
            Complex::new(0.00000000000003757931582444201,0.12776982465424247),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.04412505783961178,-0.000000000002141772606667922),
            Complex::new(0.2292216873494613,-0.25335351299684267),
            Complex::new(-0.025217900788819813,0.14795357549289817),
            Complex::new(-0.10453840778725894,-0.27891381021639533),
            Complex::new(0.22922168734686454,0.2533535130015547),
            Complex::new(-0.2798978231204447,-0.000000000005862630275442133),
            Complex::new(0.024114063582822302,0.2792581363857155),
            Complex::new(0.08237523064876884,-0.07156609038130901),
            Complex::new(-0.02521790078958657,-0.147953575492681),
            Complex::new(0.024114063582415902,-0.27925813638575625),
            Complex::new(0.3915820528594073,0.000000000004370847147419809),
            Complex::new(0.10925064034058611,0.08491387071889214),
            Complex::new(-0.10453840779597429,0.2789138102149959),
            Complex::new(0.08237523064365981,0.07156609037530123),
            Complex::new(0.10925064034972747,-0.08491387072262589),
            Complex::new(-0.30195633973889485,-0.0000000000008030062404678681),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.000000000000506283738584296,0.07053288699862277),
            Complex::new(-0.19842893769177655,-0.22779811991183496),
            Complex::new(0.07939039050695365,-0.004910413183413649),
            Complex::new(-0.3438933927736367,0.10572999104930458),
            Complex::new(0.1984289376935942,-0.22779811991018295),
            Complex::new(-0.0000000000016141242994111607,0.43010812109494856),
            Complex::new(0.11503051786126747,-0.04559906038737155),
            Complex::new(-0.09377397680025855,0.08786524284094745),
            Complex::new(-0.07939039050704318,-0.004910413183979982),
            Complex::new(-0.1150305178612396,-0.04559906038867398),
            Complex::new(0.0000000000017669538503688626,-0.35185429315125116),
            Complex::new(0.249789609473153,0.010566146631925339),
            Complex::new(0.3438933927728913,0.10572999105201959),
            Complex::new(0.09377397679963256,0.08786524284237451),
            Complex::new(-0.24978960947336515,0.010566146629960145),
            Complex::new(-0.0000000000008523717396200381,0.20856279709376593),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.18564957833729856,0.000000000004426888195872312),
            Complex::new(0.18321996035087718,-0.2928814619680452),
            Complex::new(-0.08141512551537769,0.07462690001944987),
            Complex::new(-0.16415181663823514,-0.20184426066721448),
            Complex::new(0.1832199603079654,0.29288146199386916),
            Complex::new(-0.1252495468570752,-0.0000000000057489847571370645),
            Complex::new(-0.12374535428958987,0.2696869005776657),
            Complex::new(0.1281780181081974,-0.16664588111623502),
            Complex::new(-0.08141512548014938,-0.07462690002486864),
            Complex::new(-0.12374535423520296,-0.26968690057225897),
            Complex::new(0.43708649803292726,0.000000000022632859826222298),
            Complex::new(0.10818890943323362,-0.0040841891722247295),
            Complex::new(-0.16415181665965045,0.20184426064992358),
            Complex::new(0.1281780181138434,0.16664588115801332),
            Complex::new(0.10818890941422112,0.004084189208164552),
            Complex::new(-0.26907824657769036,-0.00000000002570752841518193),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.0000000000035606207674442926,-0.2542739843765282),
            Complex::new(-0.1912963386341947,-0.06573586192396484),
            Complex::new(0.019897028310418414,0.1582744498471916),
            Complex::new(0.1078320733127061,-0.046320694030961944),
            Complex::new(0.19129633884724756,-0.06573586160629696),
            Complex::new(-0.00000000008238518004585122,-0.04560164282073404),
            Complex::new(0.33345502349805495,-0.031118687376791587),
            Complex::new(0.3215156053374418,-0.17564312863418816),
            Complex::new(-0.019897028354335152,0.1582744496396773),
            Complex::new(-0.3334550234887075,-0.031118687729938473),
            Complex::new(0.00000000019217368755109917,0.04885676404873142),
            Complex::new(-0.0558410571106585,-0.29811403580970713),
            Complex::new(-0.1078320734583722,-0.04632069381556139),
            Complex::new(-0.321515605115885,-0.1756431286129942),
            Complex::new(0.0558410573149151,-0.2981140357590023),
            Complex::new(-0.00000000017990074785060938,0.3063656466543045),
            ]));
        vector_of_cmat.push(CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.09182059496949972,0.00000000034430219584600455),
            Complex::new(-0.15648237512064864,0.2696499862508626),
            Complex::new(0.07833490729077086,-0.16681135994565893),
            Complex::new(-0.013270584264053618,0.18913128497056336),
            Complex::new(-0.15648237519714486,-0.269649986361834),
            Complex::new(0.3366471740274394,0.00000000016768580082142894),
            Complex::new(-0.05875912053940307,-0.29030462568434473),
            Complex::new(-0.16970295788387194,-0.06242819877850788),
            Complex::new(0.07833490720174995,0.166811359677963),
            Complex::new(-0.05875912017421031,0.2903046258188063),
            Complex::new(-0.38787953417627935,-0.0000000003088095109169982),
            Complex::new(-0.21547936995030423,-0.08017207782623872),
            Complex::new(-0.01327058374535493,-0.18913128471495994),
            Complex::new(-0.16970295704423854,0.06242819902211246),
            Complex::new(-0.2154793701151839,0.08017207832919451),
            Complex::new(0.21680346118132382,-0.0000000001518923541964334),
        ]));
        let target = CMatrix2::from_vec(row,col,vec![
            Complex::new(-0.11585316479558641,0.00000000000008537615059367454),
            Complex::new(-0.01617516158994864,-0.3069133890767165),
            Complex::new(0.09151776067475031,0.2893488391808999),
            Complex::new(-0.26776326422011365,-0.07615140245540597),
            Complex::new(-0.01617516158957323,0.3069133890767162),
            Complex::new(-0.5196113700126025,0.00000000000030520030946945553),
            Complex::new(0.034994249278032435,-0.7784187188424561),
            Complex::new(0.5499715689507187,0.09696990201259745),
            Complex::new(0.09151776067440744,-0.28934883918102083),
            Complex::new(0.03499424927891098,0.7784187188424454),
            Complex::new(-0.360705312722683,0.00000000000017641443861293737),
            Complex::new(-0.24964464783034968,0.07931603794662459),
            Complex::new(-0.2677632642200389,0.07615140245568816),
            Complex::new(0.5499715689506823,-0.0969699020132154),
            Complex::new(-0.24964464783047455,-0.0793160379463037),
            Complex::new(0.996169847530872,-0.0000000000005669908986760674),
            ]);

    // let gs_base = make_new_base(target, &vector_of_cmat);
    let gs_base = gs_system(target, &vector_of_cmat);

    println!("gs_base {:.10}",gs_base.adjoint());
    assert!(is_system_vertical(&gs_base, &vector_of_cmat),"base and old is not vertial.");

    }

    fn normalize(c:&CMatrix2)->CMatrix2{
        let normalizer = Complex64::new(1.0 / c.norm(),0.0);
        c * normalizer
    }

    fn is_vertical(c1:&CMatrix2,c2:&CMatrix2)->bool{
        let result = c1.dot(&c2);
        result.abs() < 1.0e-9
    }

    pub fn is_system_vertical(target: &CMatrix2, system: &Vec<CMatrix2>)->bool{
        for system_mat in system{
            if ! is_vertical(target, system_mat){
                return false;
            }
        }
        true
    }

    fn is_close(c1:&CMatrix2,c2:&CMatrix2)->bool{
        (c1 - c2).norm() < 1.0e-7
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

    // #[test]
    // fn make_new_base_test2(){
    //     let avec = vec![
    //         Complex::new(0.4,0.0),
    //         Complex::new(-0.9,0.4),
    //         Complex::new(-0.2,1.0),
    //         Complex::new(-0.9,-0.4),
    //         Complex::new(-0.6,0.0),
    //         Complex::new(0.3,0.2),
    //         Complex::new(-0.2,-1.0),
    //         Complex::new(0.3,-0.2),
    //         Complex::new(0.5,0.0)];
    //     let bvec = vec![
    //             Complex::new(-0.9,0.0),
    //             Complex::new(0.9,0.9),
    //             Complex::new(-0.3,1.0),
    //             Complex::new(0.9,-0.9),
    //             Complex::new(0.5,0.0),
    //             Complex::new(0.7,0.3),
    //             Complex::new(-0.3,-1.0),
    //             Complex::new(0.7,-0.3),
    //             Complex::new(0.5,0.0)];
    //     let a = DMatrix::from_vec(3,3,avec);
    //     let cl_a = a.clone();
    //     let re_a = reshape(a.clone(), 9, 1);
    //     let b = DMatrix::from_vec(3,3,bvec);
    //     let cl_b = b.clone();
    //     let re_b = reshape(b.clone(), 9, 1);
    //     let resultvec = vec![
    //         Complex::new(-13.03793103,0.0),
    //         Complex::new(12.53793103,13.83793103),
    //         Complex::new(-4.67931034,15.93103448),
    //         Complex::new(12.53793103,-13.83793103),
    //         Complex::new(6.86551724, 0.0),
    //         Complex::new(10.75172414,4.67931034),
    //         Complex::new(-4.67931034,-15.93103448),
    //         Complex::new(10.75172414,-4.67931034),
    //         Complex::new(7.96551724,0.0)];
    //     let minus = make_minus(&a, &b);
    //     let re_minus = reshape(minus, 9, 1);

    //     let result = DMatrix::from_vec(9, 1, resultvec);
    //     let calc_result = make_new_base(a, &vec![b]);
    //     let reshaped_result = reshape(calc_result, 9, 1);
    //     assert!(
    //         (result.clone() - reshaped_result.clone()).norm() < 1.0e-5,
    //         "result ({},{}) {:}, calc_result ({},{}) {:}, a {:}, b {:}, <a,b> {:}, <b,b> {:}, <a,b>/<b,b> {:},\n <a,b>/<b,b>b {:},a-<a,b>/<b,b>b {:},cl a-<a,b>/<b,b>b {:}, minus {:}, distane {}",
    //         result.ncols(),
    //         result.nrows(),
    //         result,reshaped_result.ncols(),
    //         reshaped_result.nrows(),
    //         reshaped_result,
    //         re_a,
    //         re_b,
    //         re_a.dot(&re_b),
    //         re_b.dot(&re_b),
    //         re_b.dot(&re_a)/ re_b.dot(&re_b),
    //         re_b.clone() * (re_b.dot(&re_a) / re_b.dot(&re_b)),
    //         re_a.clone() - (re_b.clone() * (re_b.dot(&re_a) / re_b.dot(&re_b)) + CMatrix2::zeros(9,1)),
    //         reshape(cl_a.clone() - (cl_b.clone() * (cl_b.dot(&cl_a) / cl_b.dot(&cl_b)) + CMatrix2::zeros(3,3)), 9, 1),
    //         re_minus,
    //         (result.clone() - reshaped_result.clone()));

    // }

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
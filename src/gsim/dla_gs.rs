/// # dla_gs.rs
/// dla を求める途中で Gram-Schmidt で直交化させていく関数

pub fn get_dla_gs(vector_of_hamiltonians: &Vec<CMatrix2>)->Vec<CMatrix2>{
    // まず入力を gs で直交化させておく．ここでハミルトニアンは後で使うので clone する
    let mut gs_vohs = gs(vector_of_hamiltonians);
    // 直交化した gs で以下のアルゴリズムを実行
    //[Algorithm]
    //1. gs_vohsの前から1つ取り出して old group に入れる．次にもう一つ取り出して new group に入れる
    //2. new group と old group の間で全パターン commutator を取り，その結果を comm group とする
    //3. new group を old group に加える．new group を空にする（append）
    //4. commutator が 0 でなく，old group と new group に対して独立だったら（ここで gs を使用） new groupに加える．これをcomm group すべてに実行
    //5. gs_vohs から一つ取り出して new group に入れて 2に戻る v_o_hs が空になったら終了処理(最後のnew で検査する)

    // 1 の前半を実行する部分
    let mut old = match gs_vohs.pop() {
        None=> return gs_vohs,
        Some(i)=> vec![i],
    };
    let mut new = Vec::new();

    while !gs_vohs.is_empty() {
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
                let gs_com = get_gs_vector(com,[new,old].concat());
                //  gram schmidt の結果が 0 なら線形従属である
                if is_zero(&gs_com){continue;}
                new.push(com);
            }
        }
    }
    old
}


#[cnf(test)]
pub mod test_dls_gs{
    #[test]
    fn test_get_dla_gs(){
        let rand1 = make_random_hermitian();
        let rand2 = make_random_hermitian();
        let vohs = vec![rand1,rand2];
        let dla = get_dla_gs(vohs);
    }
}

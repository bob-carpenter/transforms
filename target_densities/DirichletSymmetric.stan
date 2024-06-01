functions {
  matrix helmert_coding(int D) {
    if(D < 2) reject("Input D must be >= 2.");

    int Dm1 = D - 1;
    row_vector[Dm1] neg_ones = rep_row_vector(-1, Dm1);
    matrix[D, Dm1] helmert_mat = append_row(to_matrix(neg_ones), diag_matrix(linspaced_vector(Dm1, 1, Dm1)));

    for (i in 2:D) {
      for (j in i:Dm1) {
        helmert_mat[i, j] = -1;
      }
    }

    return helmert_mat;
  }

  matrix make_v_fullrank(matrix helmert_mat) {
    int Dm1 = cols(helmert_mat);
    int D = rows(helmert_mat);
    row_vector[D] final_row;

    if(D - 1 != Dm1) reject("Matrix input must be size D x D - 1.");

    matrix[Dm1, D] V;

    for (i in 1:Dm1) {
      V[i] = helmert_mat[, i]' / norm2(helmert_mat[, i]);
      final_row[i] = 0;
    }

    final_row[D] = 1;

   return append_row(V, final_row);
  }

  matrix make_vinv(matrix v) {
    int D = rows(v);

    if (D != cols(v)) reject("Rows and columns of input matrix must be equal.");

    return inverse(v)[1:D - 1, 1:D - 1];
  }

  matrix construct_vinv (int N) {
    return make_vinv(make_v_fullrank(helmert_coding(N)));
  }

  real target_density_lp(vector x, vector alpha){
    return dirichlet_lpdf(x | alpha);
}
}

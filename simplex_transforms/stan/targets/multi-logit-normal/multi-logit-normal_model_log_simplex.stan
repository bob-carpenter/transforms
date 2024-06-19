model {
  target += log_multi_logit_normal_cholesky_lpdf(log_x | mu, L_Sigma);
}

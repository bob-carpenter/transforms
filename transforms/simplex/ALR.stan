data {
 int<lower=0> N;
 vector<lower=0>[N] alpha;
}
parameters {
 vector[N - 1] y;
}
transformed parameters {
 simplex[N] x = softmax(append_row(y,0));
}
model {
 target += sum(y) - N * log_sum_exp(append_row(y, 0));
 target += target_density_lp(x, alpha);
}

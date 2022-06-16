data {
 int<lower=0> K;
}
parameters {
 vector[K-1] x_unc;
}
transformed parameters {
 simplex[K] x = softmax(append_row(x_unc,0));
}
model {
 target += log(exp(sum(x_unc)));
 target += -K * log1p(sum(exp(x_unc)));
}

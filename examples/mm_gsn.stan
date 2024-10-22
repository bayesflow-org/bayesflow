data {
  int<lower=1> D;  // number of observations and designs
  vector[D] designs;  // design points
  vector[D] y;  // observed outcomes
  real<lower=0> s;  // s parameter
  real<lower=0> obs_noise;
}

transformed data {
  vector[D] design_s;
  for (i in 1:D) {
    design_s[i] = (400.0 * inv_logit(designs[i]))^s;
  }
}

parameters {
  real theta1_centered;  // corresponds to theta[0] - 0.5
  real theta2_centered;  // corresponds to theta[1] - 0.5
}

transformed parameters {
  real theta1 = theta1_centered + 0.5;
  real theta2 = theta2_centered + 0.5;
  real theta1_scaled = (theta1 * (200.0-20.0) + 20.0);
  real theta2_scaled = (theta2 * (200.0-20.0) + 20.0)^s;
}

model {
  vector[D] mu;
  
  // Priors
  theta1_centered ~ normal(0.0, 0.1);
  theta2_centered ~ normal(0.0, 0.1);
  
  // Likelihood
  for (i in 1:D) {
    mu[i] = theta1_scaled * design_s[i] / (theta2_scaled + design_s[i]);
  }
  y ~ normal(mu, obs_noise);
}

generated quantities {
  vector[D] y_pred;
  for (i in 1:D) {
    y_pred[i] = normal_rng(theta1_scaled * design_s[i] / (theta2_scaled + design_s[i]), obs_noise);
  }
}

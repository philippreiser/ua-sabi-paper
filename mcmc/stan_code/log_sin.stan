functions {
    vector log_sin(vector w, vector x) {
        return w .* log(x) + 0.01 * x + 1 + sin(0.05 * x);
    }
}

data {
  int<lower=1> N_real;      // Number of measurements per w_real
  vector[N_real] y_real;     // Output exp (measured) variables
  real sigma_real; // sigma for measurement noise
  vector[N_real] x_real;       // Known input experimental variables

  real w_lower_bound;
  real w_upper_bound;
}

parameters {
  real w_real;  // unkown parameter
}

model {
  target += normal_lpdf(w_real | 1, 0.2);
  target += normal_lpdf(y_real | log_sin(rep_vector(w_real, N_real), x_real), sigma_real);
}
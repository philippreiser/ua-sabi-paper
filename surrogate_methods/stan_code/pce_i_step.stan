#include pce_helpers.stan

data {
  int<lower=1> N_real;      // Number of measurements per w_real
  int<lower=1> M;               // dimension of input variable
  int<lower=1> L;               // dimension of known input variable
  vector[N_real] y_real;     // Output exp (measured) variables
  int<lower=1> d;               // Degree of polynomials
  matrix[N_real, L] x_real;       // Known input experimental variables
  array[L] int x_idxs;           // which dimensions of input are x
  array[M-L] int w_idxs;           // which dimensions of input are

  matrix[d+1, d+1] l_poly_coeffs; // coefficients of legendrepolynomials
  int<lower=1> N_comb; // Number of selected polynomials
  array[N_comb, M] int comb; // polynomial selection
  real c_0;                                      // coefficient for constant "polynomial"
  vector[N_comb] c;                                   // coefficients of non-constant polynomials
  real sigma_approx; // sigma from surrogate training
  real sigma_real; // sigma for measurement noise

  vector[M-L] w_lower_bound;
  vector[M-L] w_upper_bound;
}

parameters {
  vector[M-L] w_real;  // estimated inputs for real (measured) variables from pce
}

model {
  target += normal_lpdf(w_real | 0, 0.5);
  matrix[N_real, M] input_real;
  input_real[:, x_idxs] = x_real;
  input_real[:, w_idxs] = rep_matrix(w_real', N_real);
  vector[N_real] mu_pred_pce_real = c_0 + get_PCE(input_real, d,
          l_poly_coeffs, comb, N_comb)*c;

  target += normal_lpdf(y_real | mu_pred_pce_real, sqrt(square(sigma_real)+square(sigma_approx)));
}
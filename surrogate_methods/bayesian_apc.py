from typing import Dict, Type
import numpy as np
import os
import cmdstanpy as csp
import arviz as az
import matplotlib.pyplot as plt

from .pce_utils import get_pce_vars, get_pce, poly_idx, get_co2_apc_vars, co2_polynomials, get_micp_apc_vars, micp_polynomials
from utils import scale_to_1
from simulation_models.co2 import CO2
from simulation_models.micp import MICP


class BayesianaPCBase:
    def __init__(self, polynomial_degree: int, log3_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, use_sparse_pce: bool, save_path: str, model: CO2|MICP, logger: Type):
        """
        Initialize the Bayesian aPC, a class to train a Bayesian PCE using MCMC via cmdstan.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log3_n_training_points (int): Log base 3 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            use_sparse_pce (bool): If True, use only selected polynomials for a sparse PCE.
            save_path (str): Path to save the fitted model.
            model (Type): Model object.
        """
        self.polynomial_degree = polynomial_degree
        self.n_samples = n_samples
        self.n_warmup = n_warmup
        self.n_chains = n_chains
        self.log3_n_training_points = log3_n_training_points
        self.prop_sigma_approx = prop_sigma_approx
        self.model = model
        self.n_total_samples = n_chains * n_samples
        self.poly_idx = poly_idx(
            polynomial_degree, model.dim_in + model.dim_param)
        self.c_dim = len(self.poly_idx) - 1
        self.save_path = save_path
        self.load_path = save_path
        if isinstance(self.model, CO2):
            self.poly = co2_polynomials()
        elif isinstance(self.model, MICP):
            self.poly = micp_polynomials(path=self.model.data_path)
        self.locations = self.model.locations
        self.times = self.model.times
        self.use_sparse_pce = use_sparse_pce
        if use_sparse_pce:
            self.idxs = np.array([[self.model.load_apc_idxs(
                polynomial_degree, location, time) for location in self.locations] for time in self.times])
            self.c_dim = len(self.idxs[0, 0])
        else:
            self.idxs = np.array(
                [[None] * len(self.locations)]*len(self.times))
        n_times = len(self.times)
        n_locations = len(self.locations)
        self.fits = [[None for _ in self.locations] for _ in self.times]
        self.c_0_samples = np.empty(
            (n_times, n_locations, self.n_total_samples, 1))
        self.c_samples = np.empty(
            (n_times, n_locations, self.n_total_samples, self.c_dim))
        self.sigma_samples = np.empty(
            (n_times, n_locations, self.n_total_samples, 1))
        self.logger = logger

    def get_training_data(self, location: int, time: int) -> Dict:
        """
        Prepare training data and pce coefficients for the cmdstan model.

        Args:
            location (int): Location index for the training data.
            time (int): Time index for the training data.

        Returns:
            Dict: A dictionary containing the training data.
        """

        train_data = self.model.load_sobol_data(
            d=self.log3_n_training_points, location=location, time=time)
        # Extract inputs and outputs from train data, assuming
        # the output is only 1-dim -> adapt for multi-dim output
        inputs = np.stack(train_data[:-1], axis=1)
        output = train_data[-1]
        assert inputs.shape[1] == self.model.dim_param + self.model.dim_in, \
            "Inputs do not match the expected dimensions of the model."

        location_idx = np.where(self.locations == location)[0][0]
        time_idx = np.where(self.times == time)[0][0]

        if isinstance(self.model, CO2):
            apc_coeffs, comb = get_co2_apc_vars(self.polynomial_degree,
                                                    (self.model.dim_param + self.model.dim_in), self.idxs[time_idx, location_idx])
        elif isinstance(self.model, MICP):
            apc_coeffs, comb = get_micp_apc_vars(self.polynomial_degree,
                                                    (self.model.dim_param + self.model.dim_in), self.idxs[time_idx, location_idx])

        training_data = {
            'M': self.model.dim_in + self.model.dim_param,
            'N': inputs.shape[0],
            'y': output,
            'input': inputs,
            'd': self.polynomial_degree,
            'a_pc_coeffs': apc_coeffs,
            'N_comb': len(comb),
            'comb': comb,
            'R2D2_mean_R2': 0.5,
            'R2D2_prec_R2': 2,
            'R2D2_cons_D2': np.ones(len(comb))*0.5
        }

        return training_data

    def train(self):
        """
        Train the Bayesian PCE model using the training data.
        """

        # Define Bayesian surrogate
        stan_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'stan_code', 'sparse_apc_t_step.stan')
        model = csp.CmdStanModel(stan_file=stan_file)
        self.stan_model = model

        for t_idx, time in enumerate(self.times):
            for l_idx, location in enumerate(self.locations):
                training_data = self.get_training_data(location, time)

                fit = model.sample(
                    data=training_data,
                    adapt_delta=0.99,
                    seed=100,
                    iter_warmup=self.n_warmup,
                    iter_sampling=self.n_samples,
                    chains=self.n_chains,
                    inits=0
                )
                self.fits[t_idx][l_idx] = fit
                self._save(fit, location, time)
                self._plot(fit, location, time)
                idata = az.from_cmdstanpy(fit)
                c_0 = np.array(idata.posterior.c_0).reshape(
                    self.n_total_samples, -1)
                c = np.array(idata.posterior.c).reshape(
                    self.n_total_samples, -1)
                sigma = np.array(idata.posterior.sigma).reshape(
                    self.n_total_samples, -1)

                self.c_0_samples[t_idx, l_idx] = c_0
                self.c_samples[t_idx, l_idx] = c
                self.sigma_samples[t_idx, l_idx] = sigma

        self.c_0_median = np.median(self.c_0_samples, axis=2)
        self.c_median = np.median(self.c_samples, axis=2)
        self.sigma_median = np.median(self.sigma_samples, axis=2)

    def load_all_fits(self):
        """
        Load all fitted cmdstan models from the specified path.
        """

        for t_idx, time in enumerate(self.times):
            for l_idx, location in enumerate(self.locations):
                self.fits[t_idx][l_idx] = self._load(
                    self.load_path, location, time)
                idata = az.from_cmdstanpy(self.fits[t_idx][l_idx])
                c_0 = np.array(idata.posterior.c_0).reshape(
                    self.n_total_samples, -1)
                c = np.array(idata.posterior.c).reshape(
                    self.n_total_samples, -1)
                sigma = np.array(idata.posterior.sigma).reshape(
                    self.n_total_samples, -1)
                self.c_0_samples[t_idx, l_idx] = c_0
                self.c_samples[t_idx, l_idx] = c
                self.sigma_samples[t_idx, l_idx] = sigma
        self.c_0_median = np.median(self.c_0_samples, axis=2)
        self.c_median = np.median(self.c_samples, axis=2)
        self.sigma_median = np.median(self.sigma_samples, axis=2)

    def _prepare_inputs(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Normalize inputs to [-1, 1] using the parameter/input ranges defined by the model.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: A two-dimensional array of shape (n, m), where 'n' is the 
            number of samples and 'm' is the model's total input dimensionality 
            ('dim_in + dim_param').
        """

        if not all(isinstance(arr, np.ndarray) and arr.ndim == 1 for arr in inputs):
            raise ValueError("All inputs must be one-dimensional np.ndarrays.")
        if len({arr.shape[0] for arr in inputs}) != 1:
            raise ValueError("All input arrays must have the same length.")
        inputs_prepared = np.column_stack(inputs)
        if inputs_prepared.shape[1] != (self.model.dim_in + self.model.dim_param):
            raise ValueError(
                "Inputs do not match the expected dimensions of the model.")
        return inputs_prepared

    def get_measurement_data(self, measurments: np.ndarray, inputs: np.ndarray) -> Dict:
        """
        Prepares the measurement data dictionary required for Stan inference.

        Args:
        measurements (np.ndarray): 
            1D array of observed measurement data.
        inputs (np.ndarray): 
            2D array of input values associated with the measurements. 
            If 'dim_in == 0', this is ignored.

        Returns:
            Dict: A dictionary containing the measurement data.
        """

        # TODO: allow for sparse PCE by vary idxs for different times & locations
        assert self.use_sparse_pce == False, "Sparse PCE not implemented yet for Parameter Inference with MCMC."
        if isinstance(self.model, CO2):
            apc_coeffs, comb = get_co2_apc_vars(self.polynomial_degree,
                                                    (self.model.dim_param + self.model.dim_in), self.idxs[0, 0])
        elif isinstance(self.model, MICP):
            apc_coeffs, comb = get_micp_apc_vars(self.polynomial_degree,
                                                    (self.model.dim_param + self.model.dim_in), self.idxs[0, 0])

        measurement_data = {
            'N_locs': len(self.locations),
            'N_times': len(self.times),
            'M': self.model.dim_in + self.model.dim_param,
            'y_real': measurments.reshape(len(self.times), len(self.locations)),
            'd': self.polynomial_degree,
            'a_pc_coeffs': apc_coeffs,
            'N_comb': self.c_dim,
            'comb': comb,
            'w_lower_bound': -1 * np.ones(self.model.dim_param),
            'w_upper_bound': np.ones(self.model.dim_param),
            'sigma_real': self.model._sigmaI.reshape(len(self.times), len(self.locations))
        }

        if self.model.dim_in > 0:
            measurement_data['L'] = self.model.dim_in
            measurement_data['x_real'] = inputs.reshape(-1, self.model.dim_in)
            measurement_data['w_idxs'] = np.arange(1, self.model.dim_param+1)
            measurement_data['x_idxs'] = np.arange(
                self.model.dim_param+1, self.model.dim_in+self.model.dim_param+1)
        if isinstance(self.model, MICP):
            measurement_data['prior_param_4_upper'] = self.model.prior_param_4_upper

        return measurement_data

    def evaluate(self, *inputs: np.ndarray) -> np.ndarray:
        """
        Abstract method for evaluation; must be implemented by subclasses.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """
        raise NotImplementedError(
            "Subclasses must implement the 'evaluate' method.")

    def _save(self, fit, location, time):
        """
        Save the fitted cmdstan model.

        Args:
            fit: The fitted cmdstan model.
            location (int): Location index for the training data.
            time (int): Time index for the training data.   
        """
        save_path = os.path.join(self.save_path, "surrogate")
        os.makedirs(save_path, exist_ok=True)
        fit.save_csvfiles(dir=os.path.join(
            save_path, f"apc_train_fit_time_{time}_loc_{location}"))

    def _load(self, load_path, location, time):
        """
        Load the fitted cmdstan model.
        """
        fit = csp.from_csv(os.path.join(
            load_path, f"surrogate/apc_train_fit_time_{time}_loc_{location}"))
        return fit

    def _plot(self, fit, location, time):
        """
        Generate and save trace plots for a fitted model.

        Args:
            fit: The fitted cmdstan model.
            location (int): Location index for the training data.
            time (int): Time index for the training data.
        """
        save_dir = os.path.join(self.save_path, f"figures/surrogate")
        os.makedirs(save_dir, exist_ok=True)

        # Convert to ArviZ InferenceData
        idata = az.from_cmdstanpy(posterior=fit)
        var_names = ["c", "c_0", "sigma"]

        # Trace plot
        trace_path = os.path.join(
            save_dir, f"trace_time_{time}_loc_{location}.png")
        az.plot_trace(idata, var_names=var_names)
        plt.tight_layout()
        plt.savefig(trace_path)
        plt.close()


class BayesianaPC(BayesianaPCBase):
    def __init__(self, polynomial_degree: int, log3_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, use_sparse_pce: bool, save_path: str, model: Type, logger: Type):
        """
        Initialize BayesianPCE.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log3_n_training_points (int): Log base 3 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            use_sparse_pce (bool): If True, use only selected polynomials for a sparse PCE.
            save_path (str): Path to save the fitted model.
            model (Type): The model object containing the sobol_sequence method and 
            the dim_in and dim_param attributes.
            logger (Type): Logger
        """

        super().__init__(polynomial_degree, log3_n_training_points, n_samples,
                         n_warmup, n_chains, prop_sigma_approx, use_sparse_pce, save_path, model, logger)

    def evaluate(self, *inputs: np.ndarray, location: int = 30, time: int = 10, random_idx: list|None = None) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by sampling one or more draws from the surrogate posterior.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.
            location (int): Location at which to evaluate the surrogate.
            time (int): Time at which to evaluate the surrogate.
            random_idx (list): Indices of surrogate posterior draws to use.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """
        inputs_prepared = self._prepare_inputs(*inputs)
        location_idx = np.where(self.locations == location)[0][0]
        time_idx = np.where(self.times == time)[0][0]
        pce_basis = get_pce(*inputs_prepared.T, p=self.polynomial_degree,
                            idx=self.idxs[time_idx, location_idx], comb=self.poly_idx, poly=self.poly)

        if random_idx is None:
            # Sample random draw from the surrogate posterior
            random_idx = np.random.randint(low=0, high=(self.n_total_samples))
        c_0_draw = self.c_0_samples[time_idx, location_idx, random_idx, :]
        c_draw = self.c_samples[time_idx, location_idx, random_idx, :]
        sigma_draw = self.sigma_samples[time_idx, location_idx, random_idx, :]

        if c_draw.ndim == 2 and c_draw.shape[0] == pce_basis.shape[0]:
            out = np.einsum("ij,ij->i", c_draw, pce_basis) + c_0_draw.squeeze()
        else:
            out = c_draw @ pce_basis.T + c_0_draw
        
        if self.prop_sigma_approx:
            out = np.random.normal(loc=out, scale=sigma_draw.squeeze())
        out = out.reshape(-1)
        return out

    def evaluate_full_posterior_predictive(self, *inputs: np.ndarray, location: int = 30, time: int = 10) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by returning the
        full posterior predictive.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """
        inputs_prepared = self._prepare_inputs(*inputs)
        location_idx = np.where(self.locations == location)[0][0]
        time_idx = np.where(self.times == time)[0][0]

        # Calculate PCE basis
        pce_basis = get_pce(*inputs_prepared.T, p=self.polynomial_degree,
                            idx=self.idxs[time_idx, location_idx], comb=self.poly_idx, poly=self.poly)
        n_points = inputs_prepared.shape[0]

        # Surrogate posterior mean prediction
        out_mean_samples = self.c_samples[time_idx,
                                          location_idx] @ pce_basis.T + self.c_0_samples[time_idx, location_idx]

        # Surrogate posterior predictive
        out_samples = np.random.normal(
            loc=out_mean_samples, scale=self.sigma_samples[time_idx, location_idx])
        return out_samples

    def evaluate_point(self, *inputs: np.ndarray, location: int = 30, time: int = 10) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by calculating the median of the posterior
        of the surrogate coefficients.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """
        inputs_prepared = self._prepare_inputs(*inputs)
        location_idx = np.where(self.locations == location)[0][0]
        time_idx = np.where(self.times == time)[0][0]
        pce_basis = get_pce(*inputs_prepared.T, p=self.polynomial_degree,
                            idx=self.idxs[time_idx, location_idx], comb=self.poly_idx, poly=self.poly)
        out = self.c_median[time_idx, location_idx] @ pce_basis.T + \
            self.c_0_median[time_idx, location_idx]
        return out


class BayesianPointaPC(BayesianaPCBase):
    def __init__(self, polynomial_degree: int, log3_n_training_points: int, n_samples: int, n_warmup: int, n_chains: int, prop_sigma_approx: bool, use_sparse_pce: bool, save_path: str, model: Type, logger: Type):
        """
        Initialize BayesianPCE.

        Args:
            polynomial_degree (int): Degree of the polynomial chaos expansion.
            log3_n_training_points (int): Log base 3 of the number of training points.
            n_samples (int): Number of samples for the MCMC.
            n_warmup (int): Number of warmup iterations for the MCMC.
            n_chains (int): Number of MCMC chains.
            prop_sigma_approx (bool): If True, the approximation error is added to the output.
            use_sparse_pce (bool): If True, use only selected polynomials for a sparse PCE.
            save_path (str): Path to save the fitted model.
            model (Type): The model object containing the sobol_sequence method and 
            the dim_in and dim_param attributes.
            logger (Type): Logger.
        """

        super().__init__(polynomial_degree, log3_n_training_points, n_samples,
                         n_warmup, n_chains, prop_sigma_approx, use_sparse_pce, save_path, model, logger)

    def evaluate(self, *inputs: np.ndarray, location: int = 30, time: int = 10, random_idx=None) -> np.ndarray:
        """
        Evaluate the Bayesian PCE by calculating the median of the posterior
        of the surrogate coefficients.

        Args:
            *inputs (np.ndarray): One or more one-dimensional arrays,
            each with shape (n,). These arrays represent the input data.

        Returns:
            np.ndarray: An array of shape (n,) with a random posterior predictive draw.
        """

        inputs_prepared = self._prepare_inputs(*inputs)
        location_idx = np.where(self.locations == location)[0][0]
        time_idx = np.where(self.times == time)[0][0]
        pce_basis = get_pce(*inputs_prepared.T, p=self.polynomial_degree,
                            idx=self.idxs[time_idx, location_idx], comb=self.poly_idx, poly=self.poly)
        out = self.c_median[time_idx, location_idx] @ pce_basis.T + \
            self.c_0_median[time_idx, location_idx]
        if self.prop_sigma_approx:
            out = np.random.normal(
                loc=out, scale=self.sigma_median[time_idx, location_idx])
        return out

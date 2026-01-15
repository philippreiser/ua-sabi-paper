from typing import Dict, List
import os
import numpy as np
import cmdstanpy as csp
import arviz as az
import concurrent.futures

from surrogate_methods.pce_utils import get_pce_vars
from surrogate_methods.bayesian_apc import BayesianaPCBase, BayesianaPC, BayesianPointaPC
from simulation_models.co2 import CO2
from simulation_models.micp import MICP
from utils import logger
import matplotlib.pyplot as plt
from datetime import datetime


class BayesianaPCMCMCBase:

    def __init__(self, mcmc_n_samples: int, mcmc_n_chains: int, mcmc_n_warmup: int, trained_pce: BayesianaPCBase):
        """
        Base MCMC Inference class for a trained Bayesian aPC.

        Args:
            mcmc_n_samples (int): Number of MCMC samples to draw from each chain.
            mcmc_n_chains (int): Number of MCMC chains.
            mcmc_n_warmup (int): Number of warmup iterations for each chain.
            trained_pce (BayesianaPCBase): A trained instance of the Bayesian PCE model.
        """

        if not hasattr(trained_pce, "fits"):
            raise ValueError(
                "Bayesian PCE model must be trained before inference.")
        self.pce = trained_pce
        self.pce_n_samples = trained_pce.n_samples
        self.pce_n_chains = trained_pce.n_chains
        self.pce_n_warmup = trained_pce.n_warmup
        self.pce_n_total_samples = trained_pce.n_total_samples
        self.mcmc_n_samples = mcmc_n_samples
        self.mcmc_n_chains = mcmc_n_chains
        self.mcmc_n_warmup = mcmc_n_warmup
        self.mcmc_n_total_samples = mcmc_n_samples * mcmc_n_chains
        self.locations = trained_pce.locations
        self.times = trained_pce.times

    def inference(self, observations: np.ndarray, inputs: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Perform Surrogate-based MCMC inference to estimate posterior distributions of unknown parameters 
        based on observed data.
        Supports batch processing for multiple observations and inputs.

        Args:
            observations (np.ndarray): Array of shape (n_datasets, n_observations) containing 
                measurement data used for inference.
            inputs (np.ndarray, optional): Array of shape (n_datasets, model.dim_input) 
                corresponding to the inputs associated with each observation set.

        Returns:
            dict: Dictionary containing MCMC samples of inferred parameters (one key for each parameter)
            - "param_1" (np.ndarray): Posterior samples of unknown parameters,
                shape (n_datasets, n_samples, model.dim_param).
            - ...
        """
        assert observations.ndim == 2, "Observations should be a 2D array (n_datasets, n_observations)."
        n_datasets = observations.shape[0]

        if inputs is not None:
            assert inputs.shape[0] == n_datasets, "Inputs should have the same dataset size as observations."

        if isinstance(self.pce.model , CO2):
            stan_file = os.path.join(os.path.dirname(__file__), "..", "surrogate_methods", "stan_code",
                                 "apc_i_step_no_input.stan" if self.pce.model.dim_in == 0 else "apc_i_step.stan")
        elif isinstance(self.pce.model, MICP):
            stan_file = os.path.join(os.path.dirname(__file__), "..", "surrogate_methods", "stan_code",
                                 "apc_i_step_no_input_micp.stan" if self.pce.model.dim_in == 0 else "apc_i_step_micp.stan")
        model = csp.CmdStanModel(stan_file=stan_file)

        param_samples_list = []
        for i in range(n_datasets):
            obs_i = observations[i]
            if inputs is not None:
                inputs_i = inputs[i]
                if inputs_i.ndim == 1:
                    inputs_i = np.expand_dims(inputs_i, axis=-1)
                inputs_i_scaled = np.column_stack(inputs_i)
            else:
                inputs_i_scaled = None
            measurement_data = self.pce.get_measurement_data(
                obs_i, inputs_i_scaled)

            self.pce.logger.info(f"Inference of Dataset {i}, Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            fit = self._inference(measurement_data, model,
                                  self.pce.prop_sigma_approx)
            self._plot(fit, i)
            idata = az.from_cmdstanpy(fit)
            param_samples = np.array(idata.posterior.w_real).reshape(
                -1, self.pce.model.dim_param)
            param_samples_list.append(param_samples)

        param_samples = np.stack(param_samples_list, axis=0)
        param_names = self.pce.model.get_param_names()
        param_dict = {name: np.expand_dims(np.moveaxis(
            param_samples, -1, 0)[i], axis=-1) for i, name in enumerate(param_names)}

        return param_dict

    def _check_mcmc_diagnostics(self, idata: az.InferenceData):
        """
        Check MCMC diagnostics: Rhat and Effective Sample Size (ESS).

        Args:
            idata (arviz.InferenceData): The inference data from MCMC.
        """

        self.pce.logger.info("Checking inference MCMC diagnostics...")

        # Rhat diagnostic
        rhat_values = az.rhat(idata, var_names=["w_real"])
        if np.any(np.array(rhat_values.w_real) > 1.01):
            self.pce.logger.warning("Rhat > 1.01 detected for w_real.")
            self.pce.logger.warning(f"w_real Rhat: {rhat_values.w_real}")

        # ESS diagnostic
        ess_values = az.ess(idata, var_names=["w_real"])
        if np.any(np.array(ess_values.w_real) < 400):
            self.pce.logger.warning("ESS < 400 detected for w_real.")
            self.pce.logger.warning(f"w_real ESS: {ess_values.w_real}")
        return np.array(rhat_values.to_array()), np.array(ess_values.to_array())

    def _plot(self, fit, dataset_idx: int, c_sample_idx: int=0):
        """
        Generate and save trace plot for a fitted model.

        Args:
            fit (CmdStanMCMC): Fitted CmdStan sampler run.
            dataset_idx (int): Index of the dataset for which the plot is generated.
        """

        save_dir = os.path.join(
            self.pce.save_path, f"figures/mcmc_parameter_inference/")
        os.makedirs(save_dir, exist_ok=True)

        # Convert to ArviZ InferenceData
        idata = az.from_cmdstanpy(posterior=fit)
        var_names = ["w_real"]

        # Trace plot
        trace_path = os.path.join(save_dir, f"trace_dataset_{dataset_idx}_c_sample_{c_sample_idx}.png")
        az.plot_trace(idata, var_names=var_names, compact=False)
        plt.tight_layout()
        plt.savefig(trace_path)
        plt.close()
    
    def _save(self, fit, dataset_idx: int, c_sample_idx: int = 0):
        """
        Save the fitted cmdstan model.

        Args:
            fit: The fitted cmdstan model.
            location (int): Location index for the training data.
            time (int): Time index for the training data.   
        """
        save_dir = os.path.join(
            self.pce.save_path, f"figures/mcmc_parameter_inference/")
        os.makedirs(save_dir, exist_ok=True)
        fit.save_csvfiles(dir=os.path.join(
            save_dir, f"parameter_inference_{dataset_idx}_loc_{c_sample_idx}"))


class BayesianaPCMCMC(BayesianaPCMCMCBase):
    def __init__(self, mcmc_n_samples: int, mcmc_n_chains: int, mcmc_n_warmup: int, trained_pce: BayesianaPC):
        """
        Initialize the MCMC inference class with a trained Bayesian PCE.

        Args:
            mcmc_n_samples (int): Number of MCMC samples to draw from each chain.
            mcmc_n_chains (int): Number of MCMC chains.
            mcmc_n_warmup (int): Number of warmup iterations for each chain.
            trained_pce (BayesianaPC): A trained instance of the Bayesian PCE model.
        """

        super().__init__(mcmc_n_samples, mcmc_n_chains, mcmc_n_warmup, trained_pce)

    def _inference(self, measurement_data: dict, model: csp.CmdStanModel, prop_sigma_approx: bool = True) -> csp.CmdStanMCMC:
        """
        Perform E-Post MCMC inference. This method runs MCMC via Stan 
        for each sample from the surrogate posterior, as described in https://arxiv.org/abs/2312.05153.

        Each single MCMC run samples from a single chain with n_samples =
        (mcmc_n_samples / pce_n_samples) * (mcmc_n_chains / pce_n_chains)

        Args:
            measurement_data (dict): Dict containing measurement data.
            model (csp.CmdStanModel): CmdStan model for surrogate-based inference.
            prop_sigma_approx (bool, optional): If True, propagates the standard deviation 
                sigma_approx is propagated to inference.
        Returns:
            fit (CmdStanMCMC): Fitted CmdStan sampler run for E-Post surrogate-based inference.
        """

        fits_csv_files = []
        single_mcmc_n_samples = int(
            (self.mcmc_n_samples / self.pce_n_samples) * (self.mcmc_n_chains) / (self.pce_n_chains))
        c_0_samples = self.pce.c_0_samples
        c_samples = self.pce.c_samples
        sigma_samples = self.pce.sigma_samples

        # Parallel execution of MCMC runs
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(self.run_mcmc, model, c_0_samples[:, :, sample_idx], c_samples[:, :, sample_idx], sigma_samples[:, :, sample_idx], measurement_data.copy(
                ), prop_sigma_approx, self.mcmc_n_warmup, single_mcmc_n_samples)
                for sample_idx in range(self.pce_n_total_samples)
            ]
            results = [future.result()
                       for future in concurrent.futures.as_completed(futures)]

        # Flatten list of CSV file paths
        fits_csv_files = [
            csv_file for result in results for csv_file in result]

        fit = csp.from_csv(fits_csv_files)
        return fit

    @staticmethod
    def run_mcmc(model: csp.CmdStanModel, c_0_sample: np.ndarray, c_sample: np.ndarray, sigma_sample: np.ndarray, measurement_data: dict, prop_sigma_approx: bool, mcmc_n_warmup: int, single_mcmc_n_samples: int) -> List[str]:
        """
        Args:
        model (csp.CmdStanModel): The compiled Stan model used for MCMC inference.
        c_0_sample (np.ndarray): A sample from the surrogate posterior for c_0.
        c_sample (np.ndarray): A sample from the surrogate posterior for c.
        sigma_sample (np.ndarray): A sample from the surrogate posterior for sigma.
        measurement_data (dict): Dictionary containing measurement data for inference.
        mcmc_settings (dict): Dictionary with MCMC configuration settings, including:
        prop_sigma_approx (bool): Whether to propagate sigma_approx to inference.
        mcmc_n_warmup (int): Number of warm-up iterations for MCMC.
        single_mcmc_n_samples (int): Number of sampling iterations.

        Returns:
            list[str]: A list of file paths to CSV output files generated by the MCMC run.
        """

        measurement_data["c_0"] = c_0_sample[:, :, 0]
        measurement_data["c"] = c_sample
        measurement_data["sigma_approx"] = sigma_sample[:, :,
                                                        0] if prop_sigma_approx else np.zeros_like(sigma_sample[:, :, 0])

        fit = model.sample(
            data=measurement_data,
            adapt_delta=0.99,
            seed=100,
            iter_warmup=mcmc_n_warmup,
            iter_sampling=single_mcmc_n_samples,
            chains=1,
            inits=0,
            show_progress=False
        )
        return fit.runset.csv_files

    def inference_e_lik(self, observations: np.ndarray, inputs: np.ndarray = None):
        """
        Perform E-Lik inference (not implemented yet).

        Args:
            observations (np.ndarray): Measurement data for inference.

        Returns:
            None (To be implemented)
        """

        assert observations.ndim == 1, "Observations should be a 1D array."
        pass  #TODO Implement E-Lik inference


class BayesianPointaPCMCMC(BayesianaPCMCMCBase):
    def __init__(self, mcmc_n_samples: int, mcmc_n_chains: int, mcmc_n_warmup: int, trained_pce: BayesianPointaPC):
        """
        Initialize the MCMC inference class with a trained Bayesian PCE.

        Args:
            mcmc_n_samples (int): Number of MCMC samples to draw from each chain.
            mcmc_n_chains (int): Number of MCMC chains.
            mcmc_n_warmup (int): Number of warmup iterations for each chain.
            trained_pce (BayesianPointPCE): A trained instance of the Bayesian Point PCE model.
        """

        super().__init__(mcmc_n_samples, mcmc_n_chains, mcmc_n_warmup, trained_pce)

    def _inference(self, measurement_data: dict, model: csp.CmdStanModel, prop_sigma_approx: bool = False) -> csp.CmdStanMCMC:
        """
        Perform Point MCMC inference via Stan. This method runs MCMC using the median of
        the surrogate posterior, as described in https://arxiv.org/abs/2312.05153.

        Args:
            measurement_data (dict): Dict containing measurement data.
            model (csp.CmdStanModel): CmdStan model for surrogate-based inference.
            prop_sigma_approx (bool, optional): If True, propagates the standard deviation 
                sigma_approx is propagated to inference.
        Returns:
            fit (CmdStanMCMC): Fitted CmdStan sampler run for Point surrogate-based inference.
        """

        measurement_data["c_0"] = self.pce.c_0_median[:, :, 0]
        measurement_data["c"] = self.pce.c_median[:, :]
        measurement_data["sigma_approx"] = self.pce.sigma_median[:, :,
                                                                 0] if prop_sigma_approx else np.zeros_like(self.pce.sigma_median[:, :, 0])

        fit = model.sample(data=measurement_data,
                           adapt_delta=0.99,
                           seed=100,
                           iter_warmup=self.mcmc_n_warmup,
                           iter_sampling=self.mcmc_n_samples,
                           chains=self.mcmc_n_chains,
                           inits={
                               "w_real": np.zeros(self.pce.model.dim_param)})
        idata = az.from_cmdstanpy(fit)
        self._check_mcmc_diagnostics(idata)
        return fit

# %%

# # # RUN FILE # # #
import numpy as np
import sys
if sys.version_info[0:2] != (3, 11):
    raise Exception("Requires Python 3.11!")
import argparse
import torch as to
import random as rd
import logging
import os
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_PLATFORMS"] = "cpu"

import importlib
import mcmc.mcmcbuilder as mcmcb
import abi.abibuilder as abib
import surrogate_methods.surrogatebuilder as sb
import simulation_models.modelloader as ml
import diagnoser as d
import visualizer as v
import utils as u

importlib.reload(u)
importlib.reload(v)
importlib.reload(d)
importlib.reload(ml)
importlib.reload(sb)
importlib.reload(abib)
importlib.reload(mcmcb)


# <parse arguments if called from terminal
def is_notebook():
    try:
        get_ipython()
        return True
    except NameError:
        return False


if is_notebook():
    config_path = os.path.join(".", "config.json")
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str,
                        required=False, help="Path to configuration file.")
    args = parser.parse_args()
    if args.config is not None:
        config_path = args.config
    else:
        config_path = os.path.join(".", "configs/micp/config_abi_low_budget.json")


# Set random seed
to.manual_seed(0)
rd.seed(0)
np.random.seed(0)


# Set number of threads
to.set_num_threads(64)


# Get config
config = u.read_config(config_path)


# Create output folder and copy config file
config.output_path = u.create_output_folders(config_path)


for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(
    os.path.join(config.output_path, "info.log"))
file_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Config read.")


# Create visualizer
vis = v.Visualizer(config)


# Get simulation model, specified in config file
logger.info("Get model...")
modelloader = ml.ModelLoader(config)
model = modelloader.get_model()

# Create and train surrogate model as specified in config file (if surrogate model needed)
if config.surrogate == "true_model":
    surrogate = model
else:
    logger.info("Build surrogate...")
    surrogatebuilder = sb.SurrogateBuilder(config, model, logger)
    surrogate = surrogatebuilder.build_surrogate()
    surrogate.train()

# Create input datasets using the true model or surrogate for training
# Create output datasets using the true model or surrogate for inference
if config.out.use_true_model:
    if config.model in ["co2", "micp"]:
        out_datasets, out_params, out_observations = model.load_abi_data(
            split="test")
    else:
        out_simulator = model.get_simulator()
        out_datasets, out_params, out_observations = model.get_simulations(
            out_simulator)
else:
    if config.model in ["co2", "micp"]:
        out_datasets, out_params, out_observations = model.load_abi_data(
            split="test", surrogate=surrogate)
    else:
        out_simulator = model.get_simulator(surrogate)
        out_datasets, out_params, out_observations = model.get_simulations(
            out_simulator)
u.save_out(config.output_path, out_datasets, out_params, out_observations)

out_datasets, out_params, out_observations = u.load_out(config.output_path)

# Create diagnoser
diag = d.Diagnoser(config)

# Perform inference
if config.inference == "abi":
    # Create and train abi model for specified model (with surrogate)
    logger.info("Build ABI...")
    abibuilder = abib.ABIBuilder(config)
    abi = abibuilder.get_abi(model, surrogate)
    if config.model in ["co2", "micp"]:
        approximator, history = abi.train_with_presimulated_data()
    else:
        simulator, approximator, history = abi.train()

    conditions = {k: v for k, v in out_datasets.items(
    ) if k not in model.get_param_names()}
    # parameter_draws = approximator.sample(conditions=conditions, num_samples=config.out.n_param_samples)
    parameter_draws = diag.measure_execution_time(
        "abi_sampling_time", approximator.sample, conditions=conditions, num_samples=config.out.n_param_samples)
    u.save_draws(config.output_path, parameter_draws)

elif config.inference == "mcmc":
    # MCMC inference for specified model (with surrogate)
    logger.info("Build MCMC Inference...")
    mcmcbuilder = mcmcb.MCMCBuilder(config, model, surrogate)
    mcmc = mcmcbuilder.build_mcmc()

    # parameter_draws = mcmc.inference(out_observations)
    if model.dim_in > 0:
        parameter_draws = diag.measure_execution_time(
            "mcmc_sampling_time", mcmc.inference, out_observations, out_datasets['inputs'])
    else:
        parameter_draws = diag.measure_execution_time(
            "mcmc_sampling_time", mcmc.inference, out_observations)
    u.save_draws(config.output_path, parameter_draws)

else:
    raise NotImplementedError("Inference type not known.")

if config.model == "micp":
    # Inference on real measurement data
    logger.info("Inference on real measurment data...")
    out_real_dataset = model.load_ct_measurements()
    if config.inference == "abi":
        conditions = {k: v for k, v in out_real_dataset.items(
        ) if k not in model.get_param_names()}
        parameter_draws_real = approximator.sample(conditions=conditions, num_samples=config.out.n_param_samples)
    elif config.inference == "mcmc":
        parameter_draws_real = mcmc.inference(out_real_dataset['observations'])
    u.save_draws(config.output_path, parameter_draws_real, "parameter_draws_real")
    vis.plot_posterior_parameter_sample(model, None, parameter_draws_real, "parameter_posterior_real")

# Diagnosis of results
logger.info("Diagnosis...")
calibration_errors = diag.measure_calibration_error(
    out_params, parameter_draws, model.get_param_names())

# Visualize details and results
logger.info("Visualize results...")

if config.surrogate == "bayesian_pce":
    vis.plot_surrogate_characteristics(surrogate)
elif config.surrogate == "bayesian_apc":
    vis.plot_apc_surrogate_characteristics(surrogate)

if model.dim_in + model.dim_param <= 2:
    if config.surrogate == "bayesian_pce":
        vis.plot_output_draws(model, surrogate)
    elif config.surrogate == "bayesian_point_pce":
        vis.plot_outputs(model, surrogate)
    elif config.surrogate == "true_model":
        vis.plot_outputs(model, None)
else:
    pass
    if config.surrogate == "bayesian_pce":
        vis.plot_output_draws_high_dim(model, surrogate)
    elif config.surrogate == "bayesian_point_pce":
        vis.plot_outputs_high_dim(model, surrogate)

if config.inference == "abi":
    vis.plot_abi_loss(history)

vis.plot_posterior_parameter_sample(model, out_params, parameter_draws)
vis.plot_posterior_parameter_recovery(model, out_datasets, parameter_draws)
vis.plot_posterior_parameter_calibration(model, out_datasets, parameter_draws)

# %%

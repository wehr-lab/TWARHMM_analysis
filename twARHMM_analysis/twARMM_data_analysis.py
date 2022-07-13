import jax.numpy as np
import jax.random as jr
import pandas as pd
import re
import pathlib as pl

from ssm.twarhmm import GaussianTWARHMM
from ssm.utils import random_rotation
from ssm.plots import gradient_cmap

import matplotlib.pyplot as plt
import seaborn as sns

#TODO: Add in data
sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink"
    ]


colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)

#%% Data read in
data = pd.read_csv("/Users/Matt/Desktop/Research/Wehr/data/0428/data_p97.csv")
#corrected_data = data.drop(columns="Unnamed: 0")
corrected_data = data.drop(columns="ID")
for column in corrected_data.keys():
    corrected_data[column] = 2*((corrected_data[column] - corrected_data[column].min())/(max(corrected_data[column]) - min(corrected_data[column]))) - 1

data_array = corrected_data.to_numpy()

#%% Prepping for HMM

# Transition matrix generation
num_states = 4
transition_probs = (np.arange(num_states) ** 10).astype(float)
transition_probs /= transition_probs.sum()
transition_matrix = np.zeros((num_states, num_states))
for k, p in enumerate(transition_probs[::-1]):
    transition_matrix += np.roll(p * np.eye(num_states), k, axis=1)

plt.imshow(transition_matrix, vmin=0, vmax=1, cmap="Greys")
plt.xlabel("next state")
plt.ylabel("current state")
plt.title("transition matrix")
plt.colorbar()
plt.show()

# Observation distributions
data_dim = 15
num_lags = 1

keys = jr.split(jr.PRNGKey(0), num_states)
angles = np.linspace(0, 2 * np.pi, num_states, endpoint=False)
theta = np.pi / 25 # rotational frequency
weights = np.array([0.8 * random_rotation(key, data_dim, theta=theta) for key in keys])
biases = np.column_stack([np.cos(angles), np.sin(angles), np.zeros((num_states, data_dim - 2))])
covariances = np.tile(0.001 * np.eye(data_dim), (num_states, 1, 1))

#%% Calculating and plotting dynamics

# Compute the stationary points
stationary_points = np.linalg.solve(np.eye(data_dim) - weights, biases)

# Plot the dynamics
if data_dim == 2:
    lim = 5
    x = np.linspace(-lim, lim, 10)
    y = np.linspace(-lim, lim, 10)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    fig, axs = plt.subplots(1, num_states, figsize=(3 * num_states, 6))
    for k in range(num_states):
        A, b = weights[k], biases[k]
        dxydt_m = xy.dot(A.T) + b - xy
        axs[k].quiver(xy[:, 0], xy[:, 1],
                      dxydt_m[:, 0], dxydt_m[:, 1],
                      color=colors[k % len(colors)])

        axs[k].set_xlabel('$x_1$')
        axs[k].set_xticks([])
        if k == 0:
            axs[k].set_ylabel("$x_2$")
        axs[k].set_yticks([])
        axs[k].set_aspect("equal")

    plt.tight_layout()
    plt.show()

#%% Initialization of time constants

time_constants = np.logspace(-1, 1, num=25, base=4)
plt.bar(np.arange(25), time_constants)
plt.ylabel(r"$\tau$")
plt.xlabel("Index")
plt.show()

#%% Creating and fitting to twARMM
key1, key2 = jr.split(jr.PRNGKey(0), 2)
test_num_states = 4  # num_states

# Initialize unspecified component randomly
twarhmm = GaussianTWARHMM(test_num_states,
                          time_constants,
                          data_dim,
                          seed=jr.PRNGKey(0))

lps, twarhmm, posteriors = twarhmm.fit(data_array)


#%% Read-in and plotting of pre-calculated states/models
def process_twarhmm_results(path:pl.Path):
    """

    Args:
        path (pathlib.Path): directory containing subdirectories of TWARHMM
        results saved out in .npy files. Can also take a string insted of a
        pathlib.Path object

    Returns:

    """
    for directory in pl.Path(path).iterdir():
    # directory = pl.Path("/Users/Matt/Desktop/Research/Wehr/talapas_shared/twARHMM_results/Tue Jun 28 15:54:00 2022")
        dir_string = str(directory)
        print("Now working in directory: {}".format(dir_string))
        for file in directory.iterdir():
            print("Scanning file: {}".format(file))
            file = str(file)
            if re.match(".*log_posteriors.npy", file):
                lps = np.load(file)
            elif re.match(".*posteriors_states.npy", file):
                expected_states = np.load(file)

        time_bins = data_array.shape[0]
        ##%% Expected posteriors
        expected_discrete_states = expected_states[0].sum(axis=2)
        prob_time_constants = expected_states[0].sum(axis=1)
        expected_time_constants = np.einsum('...i,i->...', prob_time_constants, time_constants)

        ##%% Permutated states
        # from ssm.utils import find_permutation
        # perm = find_permutation(np.argmax(expected_discrete_states, axis=1), true_states)

        ##%%
        plot_slice = (0, 1000)
        plt.figure(figsize=(12, 8))

        plt.subplot(211)
        # plt.imshow(expected_discrete_states[None,: :], aspect="auto", cmap=cmap, vmin=0, vmax=len(colors)-1)
        plt.plot(lps)
        plt.xlabel("iteration number")
        plt.ylabel("Log posterior probability")
        plt.title("LPPs")
        plt.subplot(212)
        plt.imshow(expected_discrete_states.T, aspect="auto", interpolation="none", cmap="Greys", vmin=0, vmax=1)
        plt.xlim(plot_slice)
        plt.ylabel("$z_{\\mathrm{inferred}}$")
        plt.yticks([])
        plt.xlabel("time")
        plt.title("Expected $z_t$ (prob)")

        plt.tight_layout()
        save_string = dir_string + "/states.png"
        print("Saving to {}".format(save_string))
        plt.savefig(save_string, format="png")
        plt.show()

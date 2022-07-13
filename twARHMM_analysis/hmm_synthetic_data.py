from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import scipy.stats as ss
import ssm


def rand_jitter(arr):
    stdev = .01 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(arr, frac):
    jitter_value = (np.random.random(len(arr))-0.5)*2*frac
    jitteredArr = arr + jitter_value
    return jitteredArr


#%% Define general variables
time_points = 20000
observations = 4
obs_labels = ["crange", "musSpeed", "crickSpeed", "azimuth"]
# obs 1 = crange, obs 2 = mouse speed, obs 3 = cricket speed, obs 4 = azimuth
global_states = 3
# state 1 = search, state 2 = pursuit, state 3 = catch
local_states = 2

#%% Define parameter space

crange_dict = {
    1: [0, 40],  # range in cm
    2: [0, 20],
    3: [0, 1],
}
mouse_speed_dict = {
    1: [0, 4],  # speed in cm/s
    2: [6, 15],
    3: [0, 1],
}
cricket_speed_dict = {
    1: [0, 2],  # speed in cm/s
    2: [6, 30],
    3: [0, 1],
}

azimuth_dict = {
    1: [-180, 180],  # angle in degrees
    2: [-15, 15],
    3: [-2, 2],
}
param_dict = {
    obs_labels[0]: crange_dict,
    obs_labels[1]: mouse_speed_dict,
    obs_labels[2]: cricket_speed_dict,
    obs_labels[3]: azimuth_dict,
}


#%% Generating values to fill table
# Initialize storage arrays for matrix columns

sim_data_dict = {
    obs_labels[0]: np.empty(time_points),
    obs_labels[1]: np.empty(time_points),
    obs_labels[2]: np.empty(time_points),
    obs_labels[3]: np.empty(time_points),
    "global_state": np.empty(time_points),
    "local_state": np.empty(time_points),
}


sim_data = pd.DataFrame(sim_data_dict)
true_frame = deepcopy(sim_data)
for label in obs_labels:
    true_frame[label+"_std"] = np.nan


for point in range(time_points):
    state = np.random.randint(0, 3) + 1
    local_state = np.random.randint(0, 2) + 1
    sim_data.at[point, "global_state"] = state
    sim_data.at[point, "local_state"] = local_state
    true_frame.at[point, "global_state"] = state
    true_frame.at[point, "local_state"] = local_state

    for i, obs in enumerate(obs_labels):
        # Create a dictionary that splits each global state into 2 local states using the mean value.
        local_state_dict = {
                1: np.sort([np.min(param_dict[obs][state]), np.mean(param_dict[obs][state])]),
                2: np.sort([np.max(param_dict[obs][state]), np.mean(param_dict[obs][state])])
            }
        # Get sample range for local state within global state and then pull a random value from that range
        sample_range = local_state_dict[local_state]
        value = np.random.normal(np.mean(sample_range), np.std(sample_range)/2, size=1)
        sim_data.at[point, obs] = value
        # Store the true mean and std around said mean
        true_frame.at[point, obs] = np.mean(sample_range)
        true_frame.at[point, obs+"_std"] = np.std(sample_range)

glob_col = deepcopy(true_frame.global_state)
loc_col = deepcopy(true_frame.local_state)
# Normalize between -1 and 1
# for column in sim_data.keys():
#     sim_data[column] = 2*((sim_data[column] - sim_data[column].min())/(max(sim_data[column]) - min(sim_data[column]))) - 1
#     true_frame[column] = 2*((true_frame[column] - true_frame[column].min())/(max(true_frame[column]) - min(true_frame[column]))) - 1
#
# for column in true_frame.iloc[:, [-4,-3,-2,-1]].keys():
#     true_frame[column] = 2 * ((true_frame[column] - true_frame[column].min()) / (
#                 max(true_frame[column]) - min(true_frame[column]))) - 1
#
# sim_data["global_state"] = glob_col
# sim_data["local_state"] = loc_col
# true_frame["global_state"] = glob_col
# true_frame["local_state"] = loc_col
#

#%% Sanity check for data
sim_global_means = sim_data.groupby("global_state").mean()

indi_means = sim_data.groupby(["global_state", "local_state"]).mean()

true_indi_groups = true_frame.groupby(["global_state", "local_state"]).mean()

#TODO: graph the true value +- STD and then plot sim on top
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')

gs = gridspec.GridSpec(2, 2)
gs.update(left=0.08, right=0.98, top=0.95, bottom=0.175, wspace=0.3, hspace=0.5)

crange_plot = plt.subplot(gs[0, 0])
musSpeed_plot = plt.subplot(gs[0, 1])
crickSpeed_plot = plt.subplot(gs[1, 0])
azimuth_plot = plt.subplot(gs[1, 1])

plots = [crange_plot, musSpeed_plot, crickSpeed_plot, azimuth_plot]
colors_true = ['red', 'blue']
colors_sim = ['orange', 'purple']

for i, label in enumerate(obs_labels):
    plot = plots[i]
    for glob_state in range(1, 4):
        for loc_state in range(1, 3):
            true_state_frame = true_frame[(true_frame.global_state == glob_state) & (true_frame.local_state == loc_state)]
            true_state_means = true_state_frame[label]
            true_state_std = true_state_frame[label+"_std"]

            sim_state_frame = sim_data[(sim_data.global_state == glob_state) & (sim_data.local_state == loc_state)]
            sim_state_values = sim_state_frame[label]

            x_vals = np.ones(len(sim_state_values))*glob_state
            plot.errorbar(glob_state, true_state_means.iloc[0], yerr=true_state_std.iloc[0], marker='x', mec=colors_true[loc_state-1])

            plot.plot(jitter(x_vals, 0.1), sim_state_values, 'o', mec=colors_sim[loc_state-1], alpha=0.3)
    plot.set_xticks([1, 2, 3])
    plot.set_ylabel(label)


crickSpeed_plot.set_xlabel("Global state")
azimuth_plot.set_xlabel("Global state")
labels = ["Local state 1", "sim data 1", "Local state 2", "sim data 2"]
plt.legend(labels, bbox_to_anchor=(1, 1.3), ncol=4)
plt.show()

save_frame = sim_data.drop(columns=["global_state", "local_state"])
save_frame.to_csv("/Users/Matt/Desktop/Research/Wehr/data/HMM_data.csv")

#%% testing HMM?

num_states = 3    # number of discrete states
observation_class = 'autoregressive'
obs_dim = 4       # dimensionality of observation
transitions = 'sticky'
kappa = 100  # self-transition probability prior. Can affect duration of behaviors found by model
AR_lags = 3  # How many previous values to ignore when deciding on auto-correlation?
iters = 30
hmm = ssm.HMM(num_states, obs_dim,
              observations=observation_class, observation_kwargs={'lags': AR_lags},
              transitions=transitions, transition_kwargs={'kappa': kappa})

hmm_lls = hmm.fit(save_frame, method="em", num_iters=iters)
Z = hmm.most_likely_states(save_frame)
Ps = hmm.expected_states(save_frame)
TM = hmm.transitions.transition_matrix

match_frame1 = deepcopy(sim_data)
match_frame1["predicted_state"] = Z
times = np.arange(iters+1)
plt.plot(times, hmm_lls)
plt.title("log likelihoods")
plt.show()
print(match_frame1.groupby(["global_state", "local_state"])["predicted_state"].mean())


# kappa = 1E6  # transition probability
# AR_lags = 3
# hmm = ssm.HMM(num_states, obs_dim,
#               observations=observation_class, observation_kwargs={'lags': AR_lags},
#               transitions=transitions, transition_kwargs={'kappa': kappa})
#
# hmm_lls = hmm.fit(save_frame, method="em", num_iters=iters)
# Z = hmm.most_likely_states(save_frame)
# Ps = hmm.expected_states(save_frame)
# TM = hmm.transitions.transition_matrix
#
# match_frame2 = deepcopy(sim_data)
# match_frame2["predicted_state"] = Z

#%% Hierarchical state finding
num_states = 6
kappa = 40  # self-transition probability prior. Can affect duration of behaviors found by model
AR_lags = 2  # How many previous values to ignore when deciding on auto-correlation?
iters = 30
hmm = ssm.HMM(num_states, obs_dim,
              observations=observation_class, observation_kwargs={'lags': AR_lags},
              transitions=transitions, transition_kwargs={'kappa': kappa})

hmm_lls = hmm.fit(save_frame, method="em", num_iters=iters)
Z = hmm.most_likely_states(save_frame)
Ps = hmm.expected_states(save_frame)
TM = hmm.transitions.transition_matrix

match_frame3 = deepcopy(sim_data)
match_frame3["predicted_state"] = Z
times = np.arange(iters+1)
plt.plot(times, hmm_lls)
plt.title("log likelihoods")
plt.show()
print(match_frame3.groupby(["global_state", "local_state"])["predicted_state"].mean())

#%%
fig = plt.gcf()
fig.clf()
fig.set_facecolor('w')

gs = gridspec.GridSpec(2, 2)
gs.update(left=0.08, right=0.98, top=0.95, bottom=0.175, wspace=0.3, hspace=0.5)

crange_plot = plt.subplot(gs[0, 0])
musSpeed_plot = plt.subplot(gs[0, 1])
crickSpeed_plot = plt.subplot(gs[1, 0])
azimuth_plot = plt.subplot(gs[1, 1])

plots = [crange_plot, musSpeed_plot, crickSpeed_plot, azimuth_plot]
pred_colors = ["red", "orange", "blue"]

for i, label in enumerate(obs_labels):
    plot = plots[i]
    for pred in range(0, 3):
        pred_state_frame = match_frame1[(match_frame1.predicted_state == pred)]
        pred_state_means = pred_state_frame[label]

        x_vals = np.ones(len(pred_state_means))*pred
        plot.scatter(jitter(x_vals, 0.1), pred_state_means, s=0.7, marker='o', c=pred_state_frame.global_state, alpha=0.3)
    plot.set_xticks([0, 1, 2])
    plot.set_ylabel(label)



crickSpeed_plot.set_xlabel("Predicted state")
azimuth_plot.set_xlabel("Predicted state")
#plt.colorbar(plot)
#plt.colorbar()
plt.show()
#%% Hierarchical state finding - establishing stronger priors
# Identify the current priors on the transition matrix and edit them for the global level
# ex: Allow movement between 1 -> 2 and 2 -> 1 and 2 -> 3, but no 1 -> 3 and no transitions off of 3
# Would preventing transitions off 3 mean that once 3 is hit once the model won't leave it?
# Does the way I generated my data not work? The states are random for order...

from copy import deepcopy
import scipy as sp
from sklearn import datasets as ds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ssm

#%% generating blobs of data
sample_size = 10000
# centers = [[-10, -10, -10, -10], [-5, -5, -5, -5], [0, 0, 0, 0], [5, 5, 5, 5], [10, 10, 10, 10], [20, 20, 20, 20]]
centers = [-10, -5, 0, 5, 10, 15]
blobs_x, blobs_y, blob_centers = ds.make_blobs(n_samples=sample_size, n_features=1, centers=3,
                                               shuffle=False, return_centers=True,
                                               cluster_std=0.5, random_state=30)
x_range = np.arange(sample_size)

for feature in range(blobs_x.shape[1]):
    plt.scatter(x_range, blobs_x[:, feature])


plt.show()

#%% storing blobs into data frame
df = pd.DataFrame()
for feature in range(blobs_x.shape[1]):
    column_name = "feature_{}".format(feature)
    df[column_name] = blobs_x[:, feature]
    df[column_name] = 2*((df[column_name] - df[column_name].min())/(max(df[column_name]) - min(df[column_name]))) - 1

df["time"] = x_range
df["state"] = blobs_y
states = np.unique(df.state)
# plt.scatter(df["time"], df["feature_1"], c=df["cluster_labels"], label=states)
#
# cb = plt.colorbar()
# loc = np.arange(0, max(states), max(states)/float(len(states)))
# cb.set_ticks(loc)
# cb.set_ticklabels(states)
#
# plt.legend()
# plt.show()

state_groups = df.groupby("state")

for state, group in state_groups:
        plt.scatter(group.time, group.feature_0, label=state)

plt.legend()
plt.show()

save_frame = df.drop(columns=["state", "time"])
save_frame.to_csv("/Users/Matt/Desktop/Research/Wehr/data/HMM_data.csv")
#%% State quantification/HMM fitting

num_states = 4    # number of discrete states
observation_class = 'autoregressive'
obs_dim = 1       # dimensionality of observation
transitions = 'sticky'
kappa = 1E6  # self-transition probability prior. Can affect duration of behaviors found by model
AR_lags = 3  # How many previous values to ignore when deciding on auto-correlation?
iters = 100
hmm = ssm.HMM(num_states, obs_dim,
              observations=observation_class, observation_kwargs={'lags': AR_lags},
              transitions=transitions, transition_kwargs={'kappa': kappa})
#hmm = ssm.HMM(num_states, obs_dim)

hmm_lls = hmm.fit(save_frame, method="em", num_iters=iters)
Z = hmm.most_likely_states(save_frame)
Ps = hmm.expected_states(save_frame)
TM = hmm.transitions.transition_matrix

match_frame1 = deepcopy(df)
match_frame1["predicted_state"] = Z
times = np.arange(iters+1)
plt.plot(times, hmm_lls)
plt.title("log likelihoods")
plt.show()
state_groups = match_frame1.groupby("predicted_state")

for state, group in state_groups:
        plt.scatter(group.time, group.feature_0, marker='o', label=state)

plt.legend()
plt.show()
print(match_frame1.groupby(["state"])["predicted_state"].mean())
print(match_frame1.groupby(["predicted_state"])["feature_0"].mean())

#%% Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import pathlib as pl

from twARHMM_analysis import video_utils as vu
from twARHMM_analysis import SETTINGS

video_to_label_pattern = 'Sky_mouse*[!labeled].mp4'

mouse = "0428"
video_root = SETTINGS.local_raw_data_dir + mouse + "/NickNick/"
directory_csv = SETTINGS.local_raw_data_dir + mouse + "/DirectoryKey.csv"
observation_csv = SETTINGS.local_raw_data_dir + mouse + "/data_p97.csv"
tw_data_root = SETTINGS.talapas_user + "wehrlab/twARHMM_results/"
mouse_processed_folder = SETTINGS.local_raw_data_dir + mouse + "/ProcessedData/"

save_folder = SETTINGS.local_processed_data_dir + "tiled_videos/"
video_fps = 60

initial_obs = pd.read_csv(observation_csv)
directory_key = pd.read_csv(directory_csv)
estimated_states_file = tw_data_root + "Tue Jun 28 16:23:56 2022/Tue Jun 28 16:23:56 2022posteriors_states.npy"
states_frame = np.load(estimated_states_file)
discrete_states = states_frame[0].sum(axis=2)  # Sum across the time constants to find best state

frame_obs = vu.append_video_frame_data(initial_obs, mouse_processed_folder, directory_key)
abbreviated_obs = frame_obs[0:len(frame_obs)//2]  # To match that we only have half the states labelled in the model
states_obs = vu.append_estimated_states(abbreviated_obs, discrete_states)
best_frames = vu.best_state_examples(states_obs)
model_states = best_frames.shape[0]

for index, state_row in best_frames.iterrows():
    i = 0
    file_ids = state_row["ID"]
    start_frames = state_row["start_frame"]
    end_frames = state_row["end_frame"]
    state = int(state_row["state"][0])
    list_of_videos = []
    for ID in file_ids:
        mouse_directory = directory_key.loc[directory_key["ID"] == ID].Directory.item()
        id_directory = str(video_root) + mouse_directory
        for file in pl.Path(id_directory).glob(video_to_label_pattern):
            video = vu.trimmed_video(file, start_frames[i], end_frames[i])
            list_of_videos.append(video)
            i += 1

    tiled_vid = vu.stack_videos(list_of_videos)

    video_save_root = pl.Path(str(save_folder) + mouse + "/" + f"{model_states}_state_model")
    video_save_root.mkdir(parents=True, exist_ok=True)  # Make sure dir exists
    video_save_name = pl.Path(str(video_save_root) +
                              f"/state_{state}.mp4")
    print(f"Now saving video to {video_save_name}")
    output = cv2.VideoWriter(video_save_name.as_posix(), cv2.VideoWriter_fourcc(*'mp4v'),
                             video_fps, (tiled_vid.shape[-2], tiled_vid.shape[-3]))

    print("Converting video from BGR to RGB and saving.")
    for frame in tqdm(range(tiled_vid.shape[0])):
        output.write(cv2.cvtColor(tiled_vid[frame], cv2.COLOR_BGR2RGB))
    output.release()

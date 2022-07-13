#%% Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from copy import deepcopy
import cv2
import ffmpeg
import pathlib as pl
import twARHMM_analysis.SETTINGS as SETTINGS

# TODO: Set up functions for tiling videos so all of the same state are viewable
# together. Have color of states change with state number. Add info to bottom of video

#%% Setting video file paths
mouse = "0428"
video_root = SETTINGS.local_raw_data_dir + mouse + "/NickNick/"
directory_csv = SETTINGS.local_raw_data_dir + mouse + "/DirectoryKey.csv"
observation_csv = SETTINGS.local_raw_data_dir + mouse + "/data_p97.csv"
tw_data_root = SETTINGS.talapas_user + "wehrlab/twARHMM_results/"
mouse_processed_folder = SETTINGS.local_raw_data_dir + mouse + "/ProcessedData/"
processed_save_folder_root = SETTINGS.local_processed_data_dir
for data_dir in pl.Path(tw_data_root).iterdir():
    if str(data_dir.name).startswith("."):
        continue
    for file in data_dir.glob("*.npy"):
        print("Scanning file: {}".format(file))
        file = str(file)
        if re.match(".*log_posteriors.npy", file):
            lps_file = deepcopy(file)
        elif re.match(".*posteriors_states.npy", file):
            states_file = deepcopy(file)

    # #%% Loading files
    observations = pd.read_csv(observation_csv)
    corrected_obs = observations[0:len(
        observations) // 2]  # Talapas could only handle about half the data for some reason. Would still need to normalize data if we want to include it
    lps = np.load(lps_file)
    expected_states = np.load(states_file)
    discrete_states = expected_states[0].sum(axis=2)
    total_states = discrete_states.shape[-1]
    directory_map = pd.read_csv(directory_csv)

    ##%% Iterate through and grab video labels
    prev_value = 1234123412341235
    frame_id = 0
    video_in_buffer = False
    for index, value in tqdm(corrected_obs["ID"].items()):
        if value != prev_value:
            frame_id = 0
            if video_in_buffer:
                video_save_root = pl.Path(processed_save_folder_root + mouse + "/" + mouse_directory +
                                          "/state_labeled_videos" + "twARHMM_{}_states/".format(total_states))
                video_save_root.mkdir(parents=True, exist_ok=True)  # Make sure dir exists
                video_save_name = pl.Path(str(video_save_root) +
                                          "/state_labeled_{}.mp4".format(total_states))
                print("Now saving video to {}".format(video_save_name))
                output = cv2.VideoWriter(video_save_name.as_posix(), cv2.VideoWriter_fourcc(*'mp4v'),
                                         fps, (video.shape[-2], video.shape[-3]))
                for frame in tqdm(range(video.shape[0])):
                    output.write(cv2.cvtColor(video[frame], cv2.COLOR_BGR2RGB))
                output.release()
            mouse_directory = directory_map.loc[directory_map["ID"] == value].Directory.item()
            id_directory = video_root + mouse_directory
            for file in pl.Path(id_directory).glob('Sky_mouse*[!labeled].mp4'):
                # Probably make below lines to "video" a function
                # Read in the alignment file to get needed frame numbers
                alignment_csv = pd.read_csv(mouse_processed_folder + mouse_directory + "/Alignment.csv")
                # We use the sky cam data, so grab specifically those and adjust frame numbers
                # since python starts at 0 and Matlab starts ay 1
                sky_row = alignment_csv.loc[alignment_csv["name"] == "Sky"]
                land_frame = int(sky_row["Land"][0] - 1)
                capture_frame = int(sky_row["TerminalCap"][
                                        0])  # Not taking one frame off the end since Nick thinks he included an extra frame for the data calculation at the end
                fps = sky_row["sampleRate"][0]
                # Load the file with ffmpeg and then crop to just the needed video range so it matches data
                input = ffmpeg.input(file)
                trimmed = input.trim(start_frame=land_frame, end_frame=capture_frame)
                trimmed_reset = trimmed.setpts('PTS-STARTPTS')
                out = ffmpeg.output(trimmed_reset, "pipe:", f="rawvideo", pix_fmt="rgb24")
                feed, _ = out.run(capture_stdout=True)
                video = (
                    np
                        .frombuffer(feed, np.uint8)
                        .reshape([-1, 1080, 1440, 3])
                )
                # After exporting the video to a numpy array, edit the first frame to include state label
                cv2.putText(video[frame_id],
                            "State {}".format(discrete_states[index].argmax()),
                            (50, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 7,
                            (255, 255, 0), thickness=7)
                frame_id += 1
                prev_value = value
                video_in_buffer = True
        elif value == prev_value:
            cv2.putText(video[frame_id],
                        "State {}".format(discrete_states[index].argmax()),
                        (50, 130), cv2.FONT_HERSHEY_COMPLEX_SMALL, 7,
                        (255, 255, 0), thickness=7)
            frame_id += 1

    if video_in_buffer:
        video_save_root = pl.Path(processed_save_folder_root + "0428/" + mouse_directory +
                                  "/state_labeled_videos" + "twARHMM_{}_states".format(total_states))
        video_save_root.mkdir(parents=True, exist_ok=True)  # Make sure dir exists
        video_save_name = pl.Path(str(video_save_root) +
                                  "/state_labeled_{}.mp4".format(total_states))
        print("Now saving video to {}".format(video_save_name))
        output = cv2.VideoWriter(video_save_name.as_posix(), cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps, (video.shape[-2], video.shape[-3]))
        for frame in tqdm(range(video.shape[0])):
            output.write(cv2.cvtColor(video[frame], cv2.COLOR_BGR2RGB))
        output.release()

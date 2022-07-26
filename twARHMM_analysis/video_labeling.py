#%% Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from copy import deepcopy
import cv2
import ffmpeg
import pathlib as pl

# TODO: Set up functions for tiling videos so all of the same state are viewable
#  together. Have color of states change with state number. Add session info to bottom of
#  video. Make into function taking paths as arguements. Allow it to iterate through
#  different mice. Allow to give a subset of dates to focus on incase you want to just
#  generate videos for new recordings on a daily basis
#  ADD IN-LINE COMMENTS FOR YOUR CODE YOU IJIT.
#%% Setting video file paths
"""
mouse = "0428"
video_root = SETTINGS.local_raw_data_dir + mouse + "/NickNick/"
directory_csv = SETTINGS.local_raw_data_dir + mouse + "/DirectoryKey.csv"
observation_csv = SETTINGS.local_raw_data_dir + mouse + "/data_p97.csv"
tw_data_root = SETTINGS.talapas_user + "wehrlab/twARHMM_results/"
# tw_data_root = SETTINGS.local_processed_data_dir + "twARHMM_results/"
mouse_processed_folder = SETTINGS.local_raw_data_dir + mouse + "/ProcessedData/"
processed_save_folder_root = SETTINGS.local_processed_data_dir
"""

def state_label_videos(video_root, directory_key, observation_csv,
                       model_results_folder, mouse_processed_folder, save_folder,
                       mouse,
                       video_to_label_pattern='Sky_mouse*[!labeled].mp4'):
    """
    Takes videos and labels each frame with the state the twARHMM predicts the
    mouse + cricket are in.
    Args:
        video_root (pathlib.Path or str): Path to folder containing all of the
        raw mouse folders which have the videos to be labeled.
        directory_key (pathlib.Path or str): Path to csv that contains mappings
        from observation ID number to video directory path
        observation_csv (pathlib.Path or str): Path to csv containing mouse
        observation data. Should be the same as the one used to fit the model.
        model_results_folder (pathlib.Path or str): Output folder from the model
        contains the log posteriors npy file and estimated states npy file to label
        the videos with.
        mouse_processed_folder (pathlib.Path or str): Path to directory containing
        processed folders of each mouse. These folders must contain the alignment.csv
        for each mouse so the video can be trimmed to match the same frames of
        the observation_csv.
        save_folder (pathlib.Path or str): Path to directory that mouse folders
        will be created in and outputs will be saved too.
        mouse (str): String of the mouse's name. Main variable to allow for iteration
        through multiple mice one at a time.
        video_to_label_pattern (regex string): String that will be used in
        pathlib.Path.glob() search to find the videos that will be labeled.
        Defaults to a pattern to recognize our sky camera videos.

    Returns:
        Videos saved to individual mouse folders generated in the save_folder
        location.

    """
    for data_dir in pl.Path(model_results_folder).iterdir():
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
        observations = pd.read_csv(str(observation_csv))
        corrected_obs = observations[0:len(
            observations) // 2]  # Talapas could only handle about half the data for some reason. Would still need to normalize data if we want to include it
        lps = np.load(lps_file)
        expected_states = np.load(states_file)
        discrete_states = expected_states[0].sum(axis=2)
        total_states = discrete_states.shape[-1]
        print("Current processing model has {} states".format(total_states))
        directory_map = pd.read_csv(str(directory_key))

        ##%% Iterate through and grab video labels
        prev_value = 1234123412341235
        frame_id = 0
        video_in_buffer = False
        for index, value in tqdm(corrected_obs["ID"].items()):
            if value != prev_value:
                frame_id = 0
                if video_in_buffer:
                    video_save_root = pl.Path(str(save_folder) + mouse + "/" + mouse_directory +
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
                id_directory = str(video_root) + mouse_directory
                for file in pl.Path(id_directory).glob(video_to_label_pattern):
                    # Probably make below lines to "video" a function
                    # Read in the alignment file to get needed frame numbers
                    alignment_csv = pd.read_csv(str(mouse_processed_folder) + mouse_directory + "/Alignment.csv")
                    # We use the sky cam data, so grab specifically those and adjust frame numbers
                    # since python starts at 0 and Matlab starts ay 1
                    sky_row = alignment_csv.loc[alignment_csv["name"] == "Sky"]
                    land_frame = int(sky_row["Land"][0] - 1)
                    capture_frame = int(sky_row["TerminalCap"][0])  # Not taking one frame off the end since Nick thinks he included an extra frame for the data calculation at the end
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
            video_save_root = pl.Path(str(save_folder) + mouse + "/" + mouse_directory +
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



#%% Function for separating videos into states.
# TODO: Add in funcitonality to use raw videos and not just pre-trimmed. IE incorportate
#  the alignment csv into this
def separate_videos(state_video_root, model_results_folder, directory_key,
                    observation_csv, save_folder):
    """
    Takes a bunch of pre_made state videos and cuts them into one video per state
    cluster. Tjis can results in hundreds of video per trial depending on how often
    the state switches.
    Args:
        state_video_root ():
        model_results_folder ():
        directory_key ():
        observation_csv ():
        save_folder ():

    Returns:

    """

    directory_map = pd.read_csv(directory_key)
    for file in pl.Path(model_results_folder).glob("*.npy"):
        print("Scanning file: {}".format(file))
        file = str(file)
        if re.match(".*log_posteriors.npy", file):
            lps_file = deepcopy(file)
        elif re.match(".*posteriors_states.npy", file):
            states_file = deepcopy(file)
    states = np.load(str(states_file))
    discrete_states = states[0].sum(axis=2)
    total_states = discrete_states.shape[-1]
    label_array = np.zeros(total_states).astype("int")
    final_states = np.array([discrete_states[i].argmax() for i in range(discrete_states.shape[-2])])
    states_frame = pd.DataFrame({"state": final_states})
    observations = pd.read_csv(str(observation_csv))
    states_frame["ID"] = observations.ID

    for folder in pl.Path(state_video_root).glob("*mouse*"):
        id_number = directory_map[directory_map["Directory"] == folder.name].ID.item()
        id_frame = states_frame[states_frame["ID"] == id_number]
        for state_dir in folder.glob("*_12_*"):
            for file in state_dir.glob("state*.mp4"):
                input = ffmpeg.input(file)
                current_state = 12313
                current_frame = 0
                epoch_start = True
                vid_start = True
                print("Now cutting up the video")
                for index, frame_state in tqdm(id_frame["state"].items()):
                    if (frame_state == current_state) & (not epoch_start):
                        current_frame += 1
                    elif frame_state != current_state:
                        epoch_start = True
                        if epoch_start & vid_start:
                            current_state = frame_state
                            start_frame = current_frame
                            vid_start = False
                            epoch_start = False
                        elif epoch_start & (not vid_start):
                            end_frame = current_frame
                            save_state_dir = save_folder + "state_{}/".format(current_state)
                            pl.Path(save_state_dir).mkdir(parents=True, exist_ok=True)
                            trimmed = input.trim(start_frame=start_frame, end_frame=end_frame)
                            trimmed_reset = trimmed.setpts('PTS-STARTPTS')
                            out = ffmpeg.output(trimmed_reset,
                                                save_state_dir + "state_{0}_{1}.mp4".format(current_state, label_array[current_state]),
                                                f="mp4")
                            ffmpeg.run(out, overwrite_output=True)
                            label_array[current_state] += 1
                            current_state = frame_state
                            current_frame += 1
                            start_frame = current_frame
                            input = ffmpeg.input(file)
                            epoch_start = False


def append_estiamted_states(observation_frame: pd.DataFrame, estimated_states: np.ndarray):
    """
    Function for adding the inferred state for each frame in a video as well as
    the probability associated with that inferred state.
    Args:
        observation_frame (pandas.DataFrame):Dataframe containing all observations
        used for training the model. Frame values will be appened to this frame.
        estimated_states (numpy.ndarray): Array containing the probabilities of
        each state being the correct one, summed across time constants. Generated
        by ssm.GaussianTWARHMM.fit.

    Returns:
        observation_frame (pandas.DataFrame): Input observation_frame with columns
        "best_state" and "state_probability" appended.

    """

    # Find best most likely state index position
    best_state = np.array([estimated_states[i].argmax() for i in range(estimated_states.shape[-2])])
    # Get probability value of each state
    state_probs = estimated_states[np.arange(estimated_states.shape[0]), best_state]

    # Append state index and value
    observation_frame["best_state"] = best_state
    observation_frame["state_probability"] = state_probs

    return observation_frame

# This function needs to take alignment data to get initial frame for each video
# and then generate the rest of the frame numbers to ultimately append to a csv

# TODO: Update the directory map docstring when there is a "final" function
def append_video_frame_data(observation_frame: pd.DataFrame, alignment_root,
                            directory_map: pd.DataFrame):
    """
    Function for adding the raw video frame for each observation (ie the frame in
    the original video where the observation would have been calculated) and the
    trimmed video frame for each observation (ie the frame number relative to the
    start of the hunt where the observations begin)

    Args:
        observation_frame (pandas.DataFrame): Dataframe containing all observations
        used for training the model. Frame values will be appened to this frame.
        alignment_root (str or pathlib.Path): Path to folder contianign all mouse
        processed folder that would contain the file Alignment.csv
        directory_map (pd.DataFrame): Dataframe containing two columns: Generated
        by BLANKETY-BLAKETY-BLANK

    Returns:
        observation_frame (pandas.DataFrame): Input observation_frame with columns
        "raw_frame" and "trimmed_frame" appended.

    """
    concat_frame = pd.DataFrame(columns=["raw_frame", "trimmed_frame"])
    for id_value in range(observation_frame["ID"].max()+1):
        sub_obs = observation_frame[observation_frame["ID"] == id_value]
        index_array = sub_obs.index.to_numpy()
        mouse_directory = directory_map.loc[directory_map["ID"] == id_value].Directory.item()
        alignment_csv = pd.read_csv(str(alignment_root) + mouse_directory + "/Alignment.csv")
        sky_row = alignment_csv.loc[alignment_csv["name"] == "Sky"]
        land_frame = int(sky_row["Land"][0] - 1)
        capture_frame = int(sky_row["TerminalCap"][0])
        raw_frames = np.arange(land_frame, capture_frame, 1)
        trimmed_frames = np.arange(0, capture_frame-land_frame, 1)
        add_frame = pd.DataFrame(columns=["raw_frame", "trimmed_frame"])
        add_frame["raw_frame"] = raw_frames
        add_frame["trimmed_frame"] = trimmed_frames
        add_frame.index = index_array
        concat_frame = pd.concat([concat_frame, add_frame])

    observation_frame = observation_frame.join(concat_frame)
    return observation_frame


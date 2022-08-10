import pandas as pd
import numpy as np
from tqdm import tqdm
import re
from copy import deepcopy
import cv2
import ffmpeg
import pathlib as pl
import time

from typing import Union

# TODO: Set up functions for tiling videos so all of the same state are viewable
#  together. Have color of states change with state number. Add session info to bottom of
#  video. Make into function taking paths as arguments. Allow it to iterate through
#  different mice. Allow to give a subset of dates to focus on incase you want to just
#  generate videos for new recordings on a daily basis
#  ADD IN-LINE COMMENTS FOR YOUR CODE YOU IJIT.


def append_estimated_states(observation_frame: pd.DataFrame,
                            estimated_states: np.ndarray) -> pd.DataFrame:
    """
    Function for adding the inferred state for each frame in a video as well as
    the probability associated with that inferred state.

    Args:
        observation_frame (pandas.DataFrame):Dataframe containing all observations
        used for training the model. Frame values will be appended to this frame.
        estimated_states (numpy.ndarray): Array containing the probabilities of
        each state being the correct one, summed across time constants. Generated
        by ssm.GaussianTWARHMM.fit.

    Returns:
        observation_frame (pandas.DataFrame): Input observation_frame with columns
        "best_state" and "state_probability" appended.

    """

    # Find the most likely state index position
    print("Matching most likely state for each observation")
    best_state = np.array([estimated_states[i].argmax() for i in tqdm(range(estimated_states.shape[-2]))])
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
                            directory_map: pd.DataFrame) -> pd.DataFrame:
    """
    Function for adding the raw video frame for each observation (ie the frame in
    the original video where the observation would have been calculated) and the
    trimmed video frame for each observation (ie the frame number relative to the
    start of the hunt where the observations begin)

    Args:
        observation_frame (pandas.DataFrame): Dataframe containing all observations
        used for training the model. Frame values will be appended to this frame.
        alignment_root (str or pathlib.Path): Path to folder containing all mouse
        processed folder that would contain the file Alignment.csv
        directory_map (pd.DataFrame): Dataframe containing two columns: Generated
        by BLANKETY-BLANKETY-BLANK

    Returns:
        observation_frame (pandas.DataFrame): Input observation_frame with columns
        "raw_frame" and "trimmed_frame" appended.

    """
    concat_frame = pd.DataFrame(columns=["raw_frame", "trimmed_frame"])
    print("Matching frames to observations")
    for id_value in tqdm(range(observation_frame["ID"].max()+1)):
        # Subsect to get just observations of a specific video
        sub_obs = observation_frame[observation_frame["ID"] == id_value]
        # Grab index to reset later
        index_array = sub_obs.index.to_numpy()
        mouse_directory = directory_map.loc[directory_map["ID"] == id_value].Directory.item()
        alignment_csv = pd.read_csv(str(alignment_root) + mouse_directory + "/Alignment.csv")
        # Get Sky video alignmnet frame information
        sky_row = alignment_csv.loc[alignment_csv["name"] == "Sky"]
        land_frame = int(sky_row["Land"][0] - 1)
        capture_frame = int(sky_row["TerminalCap"][0])
        # Frames for uncut video
        raw_frames = np.arange(land_frame, capture_frame, 1)
        # Frame numbers adjusted for trimmed video indexing
        trimmed_frames = np.arange(0, capture_frame-land_frame, 1)
        add_frame = pd.DataFrame(columns=["raw_frame", "trimmed_frame"])
        add_frame["raw_frame"] = raw_frames
        add_frame["trimmed_frame"] = trimmed_frames
        # Reassigning index to match original slice for concat purposes when merging
        # the two data frames
        add_frame.index = index_array
        concat_frame = pd.concat([concat_frame, add_frame])

    observation_frame = observation_frame.join(concat_frame)
    return observation_frame


def best_state_examples(observation_frame: pd.DataFrame, min_duration: int = 20)\
        -> pd.DataFrame:
    """
    Funciton for identifying 9 of the highest confidence predictions of each state
    to create a tiled video from.

    Args:
        observation_frame (pandas.DataFrame): Dataframe that contains columns
        for ID, frame numbers, estimated state, and state probability.
        min_duration (int): Minimum number of frames for the state to be
        considered valid. Defaults to 20 frames

    Returns:
        best_df (pandas.DataFrame): Dataframe with dimensions # of states*9
        rows by 5 columns.

    """

    best_df = pd.DataFrame(columns=["ID", "start_frame", "end_frame", "state", "state_probability"])
    num_states = len(observation_frame["best_state"].unique())
    print("Allocating space for dataframe")
    for state in tqdm(range(num_states)):
        # Initialize empty arrays to be filled later
        best_df.at[state, "ID"] = np.zeros(9)
        best_df.at[state, "start_frame"] = np.zeros(9)
        best_df.at[state, "end_frame"] = np.zeros(9)
        best_df.at[state, "state"] = np.zeros(9)
        best_df.at[state, "state_probability"] = np.zeros(9)

    state_start = 0
    previous_state = 0
    previous_prob = 0
    previous_frame = 0

    print("Finding best examples to tile videos with")
    time.sleep(1)
    for index, row in tqdm(observation_frame.iterrows()):
        current_state = row["best_state"]
        current_prob = row["state_probability"]
        current_ID = row["ID"]
        current_frame = row["raw_frame"]

        if current_state == previous_state:
            new_state = False
            if current_prob > previous_prob:
                previous_prob = current_prob
            previous_frame = current_frame
            previous_state = current_state
            previous_ID = current_ID
        elif current_state != previous_state:
            new_state = True

        if new_state:
            if previous_prob > best_df["state_probability"][previous_state].min():
                frame_dif = previous_frame - state_start
                if frame_dif > min_duration:
                    min_index = best_df["state_probability"][previous_state].argmin()
                    best_df["state_probability"][previous_state][min_index] = previous_prob
                    best_df["start_frame"][previous_state][min_index] = state_start
                    best_df["end_frame"][previous_state][min_index] = previous_frame
                    best_df["ID"][previous_state][min_index] = previous_ID
                    best_df["state"][previous_state][min_index] = previous_state
                    state_start = current_frame
                    previous_frame = current_frame
                    previous_prob = current_prob
                    previous_state = current_state
                    previous_ID = current_ID
                    new_state = False
                else:
                    state_start = current_frame
                    previous_frame = current_frame
                    previous_prob = current_prob
                    previous_state = current_state
                    previous_ID = current_ID
                    new_state = False
            else:
                state_start = current_frame
                previous_frame = current_frame
                previous_prob = current_prob
                previous_state = current_state
                previous_ID = current_ID
                new_state = False

    return best_df

# TODO: Create function to tile videos.
#  Inputs: Dataframe containing - ID, Start frame, End frame, State, State Probability
#  Ideas on approach: Take all videos, trim, and pipe them into np arrays. These arrays
#  can then be stacked. Take first three and hstack them, next three hstack, final three
#  hstack, then vstack the hstacks. Then use openCV to scale them down to a 1920x1080 res
#  for final export at 30 or 60 hz.
#  Design theory: One function for trimming videos since it seems to be somehting I repeat
#  One function for creating stacks, one function for resizing. Also need a function for blanks
#  Resources: Tiling - https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/
#  Scaling - https://www.codingforentrepreneurs.com/blog/open-cv-python-change-video-resolution-or-scale/


def create_blank(width: int = 1920, height: int = 1080, rgb_color: tuple = (0, 0, 0)):
    """
    Create new image(numpy array) filled with certain color in BGR

    Args:
        width (int): Width of blank image in pixels
        height (int): Height of blank image in pixels
        rgb_color (tuple): tuple of RGB values for creating color of blank
         image

    Returns:
        image (numpy.ndarray): numpy array with height rows x width columns.
         Values are BGR color values as is standard for openCV images.

    """
    # Create blank image
    image = np.zeros((height, width, 3), np.uint8)

    # Convert color to BGR for OpenCV operations
    color = tuple(reversed(rgb_color))

    # Fill image with color
    image[:] = color

    return image


def trimmed_video(file: Union[str, pl.Path], start_frame: int, end_frame: int,
                  resolution: tuple = (1440, 1080)) -> np.ndarray:
    """
    Takes input file, trims the duration to the given start and end frame, and
    then pipes it into a buffer to returned as an np.ndarray

    Args:
        file (str or pathlib.Path): The absolute file path for ffmpeg to read in
        start_frame (int): first frame of final video
        end_frame (int): final frame of final video
        resolution (tuple): Resolution as int values in (horizontal, vertical)
         order: (x, y)

    Returns:
        video (numpy.ndarray): Numpy array with rows = vertical resolution and
         columns = horizontal resolution. Each value is a BGR tuple for use in
         OpenCV
    """
    input_file = ffmpeg.input(file)
    trimmed = input_file.trim(start_frame=start_frame, end_frame=end_frame)
    trimmed_reset = trimmed.setpts('PTS-STARTPTS')
    out = ffmpeg.output(trimmed_reset, "pipe:", f="rawvideo", pix_fmt="rgb24")
    feed, _ = out.run(capture_stdout=True)
    video = (
        np
        .frombuffer(feed, np.uint8)
        .reshape([-1, resolution[1], resolution[0], 3])
    )
    return video


def loop_videos(list_of_videos):
    max_length = 0
    for vid in list_of_videos:
        if vid.shape[0] > max_length:
            max_length = vid.shape[0]
    for i, vid in enumerate(list_of_videos):
        initial_length = vid.shape[0]
        while vid.shape[0] < max_length:
            diff = max_length - vid.shape[0]
            # Either append the entire video or just the difference to loop, whichever is shorter
            append_vid = vid[0:min(diff, initial_length), :, :, :]
            vid = np.concatenate((vid, append_vid), axis=0)
        list_of_videos[i] = vid
    return list_of_videos, max_length


def stack_videos(list_of_videos: list):
    """

    Args:
        list_of_videos (list): List of 4-dimensional numpy.ndarrays with dimensions
         (time/frame info, vertical resolution, horizontal resolution, color)

    Returns:
        final_vid (numpy.ndarray): 4-dimensional array with the following properties/order:
         (time/frame info, vertical resolution, horizontal resolution, color)

    """
    num_vids = len(list_of_videos)
    vid_dim = list_of_videos[0].shape
    if num_vids < 2:
        raise IndexError("Number of videos must be two or more to make a stack!")
    if num_vids > 9:
        raise IndexError("Number of videos must be less than or equal to 9 to tile reasonably!")
    list_of_videos, max_length = loop_videos(list_of_videos)  # Loop videos to match length to max
    first_stack = list_of_videos[0:min(3, num_vids)]
    if len(first_stack) < 3:
        sec = False
        third = False
        final_vid = np.concatenate(first_stack, axis=2)  # Horizontally stacks videos
    elif num_vids == 3:
        sec = False
        third = False
        final_vid = np.concatenate(first_stack, axis=2)  # Horizontally stacks videos
    else:
        sec = True
        third = False

    if sec and num_vids > 3:
        second_stack = list_of_videos[3:min(6, num_vids)]
        if len(second_stack) < 3:
            third = False
            # Add in a blank frame so that the 3x2 array of videos stays aligned
            blank_frame = create_blank(vid_dim[2], vid_dim[1])
            blank_vid = np.stack(max_length * [blank_frame])
            second_stack.append(blank_vid)
            stack1 = np.concatenate(first_stack, axis=2)  # Horizontally stacks videos
            stack2 = np.concatenate(second_stack, axis=2)  # Horizontally stacks videos
            final_vid = np.concatenate((stack1, stack2), axis=1)  # Vertically stacks videos
        elif num_vids == 6:
            third = False
            stack1 = np.concatenate(first_stack, axis=2)  # Horizontally stacks videos
            stack2 = np.concatenate(second_stack, axis=2)  # Horizontally stacks videos
            final_vid = np.concatenate((stack1, stack2), axis=1)  # Horizontally stacks videos
        else:
            third = True

    if third and num_vids > 6:
        third_stack = list_of_videos[6:]
        if len(third_stack) < 3:
            # Add in blank frame so that 3x3 array of videos stays aligned
            blank_frame = create_blank(vid_dim[2], vid_dim[1])
            blank_vid = np.stack(max_length * [blank_frame])
            third_stack.append(blank_vid)
        stack1 = np.concatenate(first_stack, axis=2)  # Horizontally stacks videos
        stack2 = np.concatenate(second_stack, axis=2)  # Horizontally stacks videos
        stack3 = np.concatenate(third_stack, axis=2)  # Horizontally stacks videos
        final_vid = np.concatenate((stack1, stack2, stack3), axis=1)  # Horizontally stacks videos

    return final_vid


def rescale_vid(vid: np.ndarray, h_fraction: float = 0.50, v_fraction: float = 0.50):
    """

    Args:
        vid (numpy.ndarray): 4-dimensional array with the following properties/order:
         (time/frame info, vertical resolution, horizontal resolution, color)
        h_fraction (float): Value by which to scale horizontal resolution up or down.
         A value of 1 would leave the dimension as is
        v_fraction (float): Value by which to scale vertical resolution up or down. A
         value of 1 would leave the dimension as is.

    Returns:
        scaled_vid (numpy.ndarray): 4-dimensional array with the following properties/order:
         (time/frame info, vertical resolution, horizontal resolution, color). Vertical and
         horizontal resolution columns will be equal to original vid dimensions multiplied
         by the fraction input of each dimension.

    """
    resized_stack = []
    new_width = int(vid.shape[2] * h_fraction)
    new_height = int(vid.shape[1] * v_fraction)
    dsize = (new_width, new_height)
    print("Rescaling video...")
    for frame in tqdm(range(vid.shape[0])):
        # I don't know which interpolation method is best. Either INTER_CUBIC or INTER_AREA
        resized_image = cv2.resize(vid[frame,:,:,:], dsize, interpolation=cv2.INTER_CUBIC)
        resized_stack.append(resized_image)
    print("Recompiling frames for export. This may take a minute...")
    scaled_vid = np.stack(np.array(resized_stack))

    return scaled_vid


def state_label_videos(video_root, directory_key, observation_csv,
                       model_results_folder, mouse_processed_folder, save_folder,
                       mouse,
                       video_to_label_pattern='Sky_mouse*[!labeled].mp4'):
    """
    Takes videos and labels each frame with the state the twARHMM predicts the
    mouse + cricket are in.

    Args:
        video_root (pathlib.Path or str): Path to folder containing all the
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
                    video = trimmed_video(file, land_frame, capture_frame)
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


# TODO: Add in functionality to use raw videos and not just pre-trimmed. IE incorporate
#  the alignment csv into this
def separate_videos(state_video_root, model_results_folder, directory_key,
                    observation_csv, save_folder):
    """
    Takes a bunch of pre_made state videos and cuts them into one video per state
    cluster. This can result in hundreds of video per trial depending on how often
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
        # Load in files from model fit output
        print("Scanning file: {}".format(file))
        file = str(file)
        if re.match(".*log_posteriors.npy", file):
            lps_file = deepcopy(file)
        elif re.match(".*posteriors_states.npy", file):
            states_file = deepcopy(file)
    states = np.load(str(states_file))
    discrete_states = states[0].sum(axis=2)  # Collapsing time constant axis so each value is overall state probability
    total_states = discrete_states.shape[-1]
    label_array = np.zeros(total_states).astype("int")
    # Find the likely state for each observation row and store in an array
    final_states = np.array([discrete_states[i].argmax() for i in range(discrete_states.shape[-2])])
    states_frame = pd.DataFrame({"state": final_states})
    observations = pd.read_csv(str(observation_csv))
    states_frame["ID"] = observations.ID

    for folder in pl.Path(state_video_root).glob("*mouse*"):
        # Use ID value to get bath
        id_number = directory_map[directory_map["Directory"] == folder.name].ID.item()
        id_frame = states_frame[states_frame["ID"] == id_number]
        # TODO: This is hard coded for the 12 state model and should be changed with an f string
        for state_dir in folder.glob("*_12_*"):
            for file in state_dir.glob("state*.mp4"):
                input_file = ffmpeg.input(file)
                current_state = 12313
                current_frame = 0
                epoch_start = True
                vid_start = True  # A catch just for the first vid
                print("Now cutting up the video")
                for index, frame_state in tqdm(id_frame["state"].items()):
                    # Check if states are the same as previous, if not, end state and save out the frame range
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
                            trimmed = input_file.trim(start_frame=start_frame, end_frame=end_frame)
                            trimmed_reset = trimmed.setpts('PTS-STARTPTS')
                            out = ffmpeg.output(trimmed_reset,
                                                save_state_dir + "state_{0}_{1}.mp4".format(current_state, label_array[current_state]),
                                                f="mp4")
                            ffmpeg.run(out, overwrite_output=True)
                            label_array[current_state] += 1
                            current_state = frame_state
                            current_frame += 1
                            start_frame = current_frame
                            input_file = ffmpeg.input(file)
                            epoch_start = False


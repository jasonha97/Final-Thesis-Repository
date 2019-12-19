# Import main libraries for data processing:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modify sys file path array to ensure it can access the scripts:
import sys
sys.path.insert(0, r"C:\Users\Jason Ha\Documents\University\4th Year\Thesis Project (Repo)\Jupyter Notebooks\Active Projects\scripts")

# Now, import custom scripts for EDF processing and bandpass filtering
from qps_extract_functions import butter_bandpass_filter, butter_bandpass
from edf_annot_extract import read_edf_annotations, set_the_annotations

# Import signal processing functions from Scipy
from scipy.signal import butter, lfilter, sosfilt, filtfilt
from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import hilbert
from scipy.signal.windows import hann

# For any power computations
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper

# Set up seaborn if plotting is performed
sns.set()

# Import libraries for reading EDF files (if necessary):
import mne

# import libraries for non-linear regression and parameter extraction
from lmfit import Parameters, minimize

# For generating a QPS waveform:
from wavelet_generator import qps_time_sig

###################################
# Function: 'read_eeg_and_stages' #
###################################

def read_eeg_and_annot(absolute_filepath, psg_rec, sleep_stage_annot, scorer_annot_1, scorer_annot_2 = None):

	"""
	Function: 'read_eeg_and_stages'

	This function accepts the raw PSG recording (in EDF format) as well as its
	corresponding hypnogram (sleep stage) annotation file and the scorer annotation
	for spindles detected by expert scorers. 

	The function uses the hypnogram annotations to create 20 second epochs for all channels
	in the form of a pandas dataframe. It should also return the raw PSG file itself (for future use),
	the intersection of the spindle annotations (if there are TWO scorers) and more (under 'returns').

	Parameters:
		- argv[0] = absolute_filepath: The filepath that contains the PSG and annotations (dtype = str)
		- argv[1] = psg_rec: The PSG recording (in EDF) in the absolute filepath (dtype = str)
		- argv[2] = sleep_stage_annot: An EDF file containing the sleep stages for the corresponding PSG (dtype = str)
		- argv[3] = scorer_annot_1: An EDF file containing the spindle annotations from scorer 1.
		- argv[4] = scorer_annot_2: An EDF file containing the spindle annotations from scorer 2. Not necessary if there is only ONE scorer.

	Returns:
		- raw, epochs_df, scorer_intersection, sampling_frequency
	"""

	#######################################################
	# Step 1: Need to load the raw PSG into the workspace #
	#######################################################

	psg_file = absolute_filepath + "\\" + psg_rec
	stage_annot = absolute_filepath + "\\" + sleep_stage_annot

	# Load raw MNE file.
	raw = mne.io.read_raw_edf(psg_file, preload=False, verbose=False) 

	###########################################################################
	# Step 2: Load sleep stage annoations onto the raw file and create epochs #
	###########################################################################

	# Load annotaions per the sleep stage EDF file into workspace. Then set annots onto raw
	stages = mne.read_annotations(stage_annot)
	raw.set_annotations(stages)

	# Do not filter the PSG recordings yet. Only do this when accessing each individual window. 
	# Perform this in the 'compute_features' function.

	# Get the event and event_id from the annotations:
	event, event_id = mne.events_from_annotations(raw)

	# Create epochs from the event and event_id
	epochs = mne.Epochs(raw, event, event_id, tmin=0.0, tmax=20, baseline=None) # 20 second epochs (based off stage annotations)

	# Create dataframe from epochs
	epochs_df = epochs.to_data_frame()

	#########################################################################
	# Step 3: Get the intersection between both scorers (if there are two): #
	#########################################################################

	if (scorer_annot_2 == None):

		# This means there is only a single scorer (scorer 1). This annotation will serve as the intersection.
		scorer_1 = absolute_filepath + "\\" + scorer_annot_1
		s1_df = pd.read_csv(scorer_1)
		s1_df.columns = ['start', 'duration']
		s1_df['end'] = s1_df['start'] + s1_df['duration']

		scorer_intersection = s1_df

	else:

		# In this case, we have TWO scorers. We need to perform an inner join to find the intersection
		# between the two scorers.

		# Begin with scorer 1:
		scorer_1 = absolute_filepath + "\\" + scorer_annot_1
		s1_df = pd.read_csv(scorer_1)
		s1_df.columns = ['start', 'duration']
		s1_df['end'] = s1_df['start'] + s1_df['duration']

		# Make a temporary column for the start times rounded to the nearest integer:
		s1_df['start_floor'] = s1_df['start'].astype('int32')

		# Then load scorer 2:
		scorer_2 = absolute_filepath + "\\" + scorer_annot_2
		s2_df = pd.read_csv(scorer_2)
		s2_df.columns = ['start', 'duration']
		s2_df['end'] = s2_df['start'] + s2_df['duration']

		# Make a temporary column for the start times rounded to the nearest integer:
		s2_df['start_floor'] = s2_df['start'].astype('int32')

		# Get the 'inner join' of the two scorers' dataframes in order to get the ground truth.
		scorer_inner = pd.merge(s1_df, s2_df, how='inner', on=['start_floor'])
		scorer_inner.head()

		# Take the mean of the start (onset) tie, duration of the spindle and the end time for each scorer:
		scorer_final = pd.DataFrame()
		scorer_final['start'] = (scorer_inner['start_x'] + scorer_inner['start_y']) / 2
		scorer_final['duration'] = (scorer_inner['duration_x'] + scorer_inner['duration_y']) / 2
		scorer_final['end'] = (scorer_inner['end_x'] + scorer_inner['end_y']) / 2

		scorer_intersection = scorer_final

	########################################################
	# Step 4: Get the sampling frequency from the raw file #
	########################################################

	sampling_frequency = float(raw.info['sfreq'])

	##############
	# Final Step #
	##############

	return (raw, epochs_df, scorer_intersection, sampling_frequency)

#########################################
# Function: 'isolate_stage_and_channel' #
#########################################

def epochs_isolate(epochs_df, sleep_stage, ch_name):

	"""
	Function: "epochs_isolate"

	'epochs_isolate' accepts the epochs derived from the 'read_eeg_and_annot' function
	and the desired sleep stage name and channel name to pull out from the dataframe.
	This function essentialy works solely with a pandas dataframe.

	Parameters:
		- epochs_df: Dataframe containing the sleep stages and epochs for each PSG channel (EEG, EOG, EMG etc.) (dtype = pandas dataframe)
		- sleep stage: The associated sleep stage desired from the dataframe. Per MASS conventions, the stages are either
		               one of the following: 1, 2, 3, 4, ?, R and W (dtype = str (from command line))
		- ch_name: The channel type and the electrode placement (10-20 system) (e.g. EEG C3-LER)

	Returns:
		- eeg_epochs: Filtered dataframe as per the inputs into the function.
		- epoch_list: List of all the epochs relevant to the sleep stage desired from user:
	"""

	def_string = "Sleep stage "
	stage_str = str(sleep_stage) # If need be:
	stage_wanted = def_string + stage_str

	#####################################################################
	# Step 1: Reset the indices for epochs_df and get the epoch numbers #
	#####################################################################

	temp = epochs_df.reset_index(drop=False)
	tmp = temp[temp['condition'] == stage_wanted]
	epoch_list = tmp['epoch'].unique()
	print("Number of Relevant Epochs = {}".format(len(epoch_list)))

	########################################################################
	# Step 3: Pull out the relevant epochs and channel from the dataframe: #
	########################################################################

	selected_channels = epochs_df[[ch_name]] # Pull out as a dataframe, not a series
	eeg_epochs = selected_channels.loc[stage_wanted, :]

	return (eeg_epochs, epoch_list)

##################################
# Function: manual_class_and_qps #
##################################

def manual_class_and_qps(eeg_epochs, epoch_list, scorer_intersection, ch_name, window_length, window_stride, sampling_rate):

	"""
	Function: 'manual_class_and_qps'

	'manual_class_and_qps' performs manual classification for spindles that have been identified by expert scorers.
	The process begins by generating a lower and upper threshold for tentative spindles. These thresholds are then used to detect
	preliminary spindles by generating an onset-offset marker pair. Note that these annotations are NOT the expert scorer annotations
	and are ONLY used to generate labels for the sake of the initialisation of the QPS parameters during the NLLS regression stage.

	The expert annotations are used to generate the OFFICIAL label for each frame acquired by the frame-acquisition process. If the
	frame OFFICIALLY does NOT pass an expert onset/offset marker, we MUST classify it as a non-spindle. The opposite is true for spindles
	detected! 

		- Uses a 1.0 sec (100 ms) window with a 0.5 sec (50 ms) overlap/stride as it moves along the entire 20.0 sec epoch
		- The moving window checks if the expert-scored annotations lies in the moving window.
			- If the window has moved past the annotation more than 50% of the windows duration (i.e. 50ms), then classify as a spindle (1)
			- Else, classify the spindle as a non-spindle (0).
		- For the spindles/non-spindles captured by the frame, perform NLLS to compute a, b, c, d, e and f
		- Also compute the other features (residual, RSER, QSER, RMS etc.) that was performed in the previous iterations of the project
		- Collect all the data and compile all in a single .csv file for the single patient.

	Parameters:
		- eeg_epochs: Desired epochs (and PSG channel) from the pre-processed epochs_df pandas DataFrame
		- epoch_list: Standalone list of epochs the epochs_df pandas DataFrame
		- scorer_intersection: DataFrame containing the onset, duration and end of each spindle annotated by one/two expert scorers.
		- ch_name: The desired PSG channel to be pulled from the eeg_epochs dataframe.
		- window_length: The length of the moving window to be used to sample (in seconds) a portion of the 20.0 second epoch.
		- window_stride: The movement of the window (in seconds) along the epoch. The stride should ideally be less than the window_length to ensure overlap occurs.
		- sampling_rate: The sampling_rate used for the recording of the raw PSG recording.

	Returns:
		- final_df: The final dataframe containing the QPS parameter values and values for all other relevant features. To be saved after the function.

	"""

	# Create variables/parameters to be used for the windowing/framing:
	Tw = 1.0							# 1.0 second window length
	stride = 0.5						# 0.5 second stride movement of the window along the epoch.
	frame_len = int(Tw * 1000 / 4)		# frame_len = Tw but in terms of indices. Multiply by 1000 to convert to ms. Divide by 4 to go from ms to samples.
	shift = int(0.5 * 1000 / 4)			# shift = stride buy in terms of indices. Mult by 1000 -> convert to ms. Divide by 4 to go from ms to samples.
	curr_shift_factor = 0				# Factor is used to multiply with the stride. This effectively move the window by the time defined by 'stride' (or 'shift')

	# Create a dataframe to collate ALL computed values and features from the process below.
	final_df = pd.DataFrame()

	# We also want another dataframe comprising ONLY of labels for each row of 'final_df':
	label_df = pd.DataFrame()

	# Completed epochs - For print out purposes on the console.
	completed_epochs = 1;	

	# Create a for-loop to access each epoch in 'epoch_list'. This is the outermost loop:
	for i in epoch_list:

		# Get the current epoch index being accessed from the list:
		epoch_idx = i
		curr_epoch = eeg_epochs.loc[i]
		signal = curr_epoch[ch_name]

		# Any 'signal' containing NaN values needs to be fixed before proceeding to the windowing stage.
		# The EEG signal can be treated as a time-series. Use pd.interpolate(method = 'time') to interpolate the missing sections in the signal:
		signal = signal.interpolate(method = 'time').fillna(0)

		print()
		print("Current Epoch Number = {}\n".format(epoch_idx))
		print("Epoch's Completed = {}/{}".format(completed_epochs, len(epoch_list)))

		completed_epochs = completed_epochs + 1

		# Get the times of each sample in 'signal' captured in the curr_epoch. The time array will ONLY be used to check for the presence of annotations:
		time = curr_epoch.index.values / 1000  	   # ms -> seconds
		time = time + (epoch_idx * 20)			   # The current time-interval can be accessed by multiplying the time by the epoch number * 20 seconds

		# Generate the annotations via the automatic detection algorithm process
		auto_intersection = auto_spindle_detect(time, signal, sampling_rate)

		# The 'curr_shift_factor' has an upper bound that is restricted by the last index of the epoch. We can calculate this upper bound before proceeding.
		csf_upper_bound = int(np.ceil((len(time) - frame_len) / shift))

		###########################################################################
		# Run 1: Feature Computation Using Annotations From Auto Detect Algorithm #
		###########################################################################

		print()
		print("##############################")
		print("# Run 1: Feature Computation #")
		print("##############################")
		print()

		# Flag to keep check if a frame has went PAST an onset but is YET to detect the OFFSET:
		# We can assume that at the VERY start, no spindle has been detected. Initialise as 1.
		annotation_finish = 1;

		# Now, we want to use a window to access 2.0 second portions of the current epoch. This window will then be shifted along the epoch progressively.
		for curr_shift_factor in range(csf_upper_bound):

			time_window = time[0 + (curr_shift_factor*shift): frame_len + (curr_shift_factor*shift)] 		# JUST TO CHECK AGAINST ANNOTATION
			signal_window = signal[0 + (curr_shift_factor*shift): frame_len + (curr_shift_factor*shift)]	# THIS IS WHAT WE CARE ABOUT.

			# Boolean condition to check if time_window contains an annotation from 'auto_intersection' DataFrame:
			on_condition = (scorer_intersection['start'] >= min(time_window)) & (scorer_intersection['start'] <= max(time_window))
			off_condition = (scorer_intersection['end'] >= min(time_window)) & (scorer_intersection['end'] <= max(time_window))

			# If this is true, we should see a particular spindle extracted from the annotations. 
			fall_in_on = scorer_intersection[on_condition]
			fall_in_off = scorer_intersection[off_condition]

			# Get the lengths of the onset and offset dataset generated. Their lengths are the basis for the manual classification.
			num_onset = len(fall_in_on)
			num_offset = len(fall_in_off)

			# Print out the current frame number:
			print("Frame Number: {}".format(curr_shift_factor))

			# A 1.0 second window can either contain ONE spindle or NONE at all. We can use a conditional block to check.
			if (annotation_finish == 1):

				# This means we have YET to detect an onset for a new spindle
				# This technically satifies having completely past an annotated spindle.
				# Now check whether or not the frame is detected an onset or not:

				if (num_onset == 1):

					# Frame captures an onset for the FIRST time OR after exiting a preceding microevent
					# Compute the percentage with which the frame has PASSED the onset. Must be >= 50%!
					elapsed_on = max(time_window) - fall_in_on['start']
					elapsed_on_percentage = float(elapsed_on / Tw * 100)

					# Need to check if the percentage elapsed is greater than or equal to 50%:
					if (elapsed_on_percentage >= 50.0):

						# If so, we should be good to classify the frame as having captured a spindle.
						# We need to check if the offset is in the frame or not and computed the percentage
						
						if (num_offset == 1):

							# Elapsed percentage ALSO for the offset marker.
							elapsed_off = max(time_window) - fall_in_off['end']
							elapsed_off_percentage = float(elapsed_off / Tw * 100)

							# Now, we need to check if the offset is <= 50%. We want it as low as possible
							# as this means the frame is well WITHIN the spindle and has not passed it considerably
							if (elapsed_off_percentage <= 50.0):

								# THERE IS A SPINDLE PRESENT IN THE WINDOW. Print to user to confirm:
								print("A tentative spindle occurs in this window! Spindle used to compute features:")

								# This means we are in the safe zone to classify the frame as having
								# captured a spindle. Update the label flag:
								s_or_ns = 1

								print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
								print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
								print("Label For Frame = {}".format(s_or_ns))
								print()

								# Ready to compute QPS parameters and features:
								feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

								# Append feature_df onto final_df:
								if (len(final_df) == 0):
									# i.e. We starting fresh.
									for col in feature_df.columns:
										final_df[col] = feature_df[col]

								else:
									# Need a temporary dataframe:
									tmp = pd.DataFrame()
									for col in feature_df.columns:
										tmp[col] = feature_df[col]

									final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

								# We have not yet passed the offset (> 50%). Hence, we have yet to walk
								# past the annotations. Toggle the annotation_finish flag to 0
								annotation_finish = 0

							else:

								# This means we most likely walked too far over the offset (greater than 50% of the frame length)
								# If this is the case, then its most likely than the onset was NEVER in the frame in the first place.
								# We should expect this condition to never be reached. Still, we classify as a non-spindle (0)
								# and toggle the annotation_finish back to 1 (since we have technically walked way past the microevent)
								s_or_ns = 0

								print("PARADOX! (TYPE-1)")
								print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
								print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
								print("Label For Frame = {}".format(s_or_ns))
								print()
								plt.show()

								# Ready to compute QPS parameters and features:
								feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

								# Append feature_df onto final_df:
								if (len(final_df) == 0):
									# i.e. We starting fresh.
									for col in feature_df.columns:
										final_df[col] = feature_df[col]

								else:
									# Need a temporary dataframe:
									tmp = pd.DataFrame()
									for col in feature_df.columns:
										tmp[col] = feature_df[col]

									final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

								annotation_finish = 1

						else:

							# Most likely means the frame has NOT captured the offset marker. 
							# We are in the safe-zone to classify the frame has having captured a spindle
							s_or_ns = 1;

							print("A tentative spindle occurs in this window! Spindle used to compute features:")
							print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
							print("NO OFFSET IN FRAME.")
							print("Label For Frame = {}".format(s_or_ns))
							print()

							# # Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
							# # whilst performing the computations.
							# fig = plt.figure(figsize=(10,5))
							# plt.plot(time_window, signal_window)
							# plt.axvline(float(fall_in_on['start']), color='red', label='Spindle Onset')
							# plt.axvspan(float(fall_in_on['start']), max(time_window), color='red', alpha=0.15)
							# plt.legend()
							# plt.show()

							# Ready to compute QPS parameters and features:
							feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

							# Append feature_df onto final_df:
							if (len(final_df) == 0):
								# i.e. We starting fresh.
								for col in feature_df.columns:
									final_df[col] = feature_df[col]

							else:
								# Need a temporary dataframe:
								tmp = pd.DataFrame()
								for col in feature_df.columns:
									tmp[col] = feature_df[col]

								final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

							# We need to toggle the 'annotation_finish' flag to 0 since we have YET to
							# reach the END of the microevent (signified by the offset marker)
							annotation_finish = 0;


					else:

						# Else, the frame has NOT sufficiently passed the onset marker. Classify as a non-spindle.
						s_or_ns = 0;

						print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
						print("NO OFFSET IN FRAME")
						print("Label For Frame = {}".format(s_or_ns))
						print()

						# Ready to compute QPS parameters and features:
						feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

						# Append feature_df onto final_df:
						if (len(final_df) == 0):
							# i.e. We starting fresh.
							for col in feature_df.columns:
								final_df[col] = feature_df[col]

						else:
							# Need a temporary dataframe:
							tmp = pd.DataFrame()
							for col in feature_df.columns:
								tmp[col] = feature_df[col]

							final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

						# Keep the annotation_finish flag at 1 since this we haven't 'officially' detected a spindle yet
						# and, hence, haven't walked PAST an onset marker.
						annotation_finish = 1;

				else:

					# This means that we ALSO haven't captured an onset. Automatically classify as a non-spindle
					s_or_ns = 0

					print("NO ONSET IN FRAME")
					print("NO OFFSET IN FRAME.")
					print("Label For Frame = {}".format(s_or_ns))
					print()

					# Ready to compute QPS parameters and features:
					feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

					# Append feature_df onto final_df:
					if (len(final_df) == 0):
						# i.e. We starting fresh.
						for col in feature_df.columns:
							final_df[col] = feature_df[col]

					else:
						# Need a temporary dataframe:
						tmp = pd.DataFrame()
						for col in feature_df.columns:
							tmp[col] = feature_df[col]

						final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

					# Keep the annotation_finish flag at 1. We still haven't detected an onset:
					annotation_finish = 1


			elif (annotation_finish == 0):

				# This means we HAVE walked past an onset but have YET to pass an offset completely
				# In this situation, it essentially means we are in the MIDDLE of a spindle event
				# and the ONSET has been passed completely BUT the offset has NOT been walked past yet
				# We can see this as a case of the frame size being considerably SMALLER than the duration
				# of the spindle.

				# The case where an onset is detected is UNLIKELY to be met in this case.
				if (num_onset == 1):

					# If an onset so HAPPENS to still be in the frame, we need to check conditions once more:
					elapsed_on = max(time_window) - fall_in_on['start']
					elapsed_on_percentage = float(elapsed_on / Tw * 100)

					if (elapsed_on_percentage >= 50.0):

						# Now, we need to check the offset marker:
						if (num_offset == 1):

							# Elapsed percentage ALSO for the offset marker.
							elapsed_off = max(time_window) - fall_in_off['end']
							elapsed_off_percentage = float(elapsed_off / Tw * 100)

							# Now, we need to check if the offset is <= 50%. We want it as low as possible
							# as this means the frame is well WITHIN the spindle and has not passed it considerably
							if (elapsed_off_percentage <= 50.0):

								# This means we are in the safe zone to classify the frame as having
								# captured a spindle. Update the label flag:
								s_or_ns = 1

								print("A tentative spindle occurs in this window! Spindle used to compute features:")
								print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
								print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
								print("Label For Frame = {}".format(s_or_ns))
								print()

								# Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
								# whilst performing the computations.
								# fig = plt.figure(figsize=(10,5))
								# plt.plot(time_window, signal_window)
								# plt.axvline(float(fall_in_on['start']), color='red', label='Spindle Onset')
								# plt.axvline(float(fall_in_off['end']), color='blue', label='Spindle End')
								# plt.axvspan(float(fall_in_on['start']), float(fall_in_off['end']), color='red', alpha=0.15)
								# plt.legend()
								# plt.show()

								# Ready to compute QPS parameters and features:
								feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

								# Append feature_df onto final_df:
								if (len(final_df) == 0):
									# i.e. We starting fresh.
									for col in feature_df.columns:
										final_df[col] = feature_df[col]

								else:
									# Need a temporary dataframe:
									tmp = pd.DataFrame()
									for col in feature_df.columns:
										tmp[col] = feature_df[col]

									final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

								# We have not yet passed the offset (> 50%). Hence, we have yet to walk
								# past the annotations. Toggle the annotation_finish flag to 0
								annotation_finish = 0

							else:

								# This means we most likely walked too far over the offset (greater than 50% of the frame length)
								# If this is the case, then its most likely than the onset was NEVER in the frame in the first place.
								# We should expect this condition to never be reached. Still, we classify as a non-spindle (0)
								# and toggle the annotation_finish back to 1 (since we have technically walked way past the microevent)
								s_or_ns = 0

								print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
								print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
								print("Label For Frame = {}".format(s_or_ns))
								print()

								# Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
								# whilst performing the computations.
								# fig = plt.figure(figsize=(10,5))
								# plt.plot(time_window, signal_window)
								# plt.axvline(float(fall_in_on['start']), color='red', label='Spindle Onset')
								# plt.axvline(float(fall_in_off['end']), color='blue', label='Spindle End')
								# plt.axvspan(float(fall_in_on['start']), float(fall_in_off['end']), color='red', alpha=0.15)
								# plt.legend()
								# plt.show()

								# Ready to compute QPS parameters and features:
								feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

								# Append feature_df onto final_df:
								if (len(final_df) == 0):
									# i.e. We starting fresh.
									for col in feature_df.columns:
										final_df[col] = feature_df[col]

								else:
									# Need a temporary dataframe:
									tmp = pd.DataFrame()
									for col in feature_df.columns:
										tmp[col] = feature_df[col]

									final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

								annotation_finish = 1

						else:

							# Perhaps we're still trudging through the event since the window stride is SLOW
							# Classify as a spindle. We have yet to walk sufficiently past the offset.
							s_or_ns = 1

							print("A tentative spindle occurs in this window! Spindle used to compute features:")
							print("Percentage Time Frame Elapsed After ONSET = {}".format(elapsed_on_percentage))
							print("NO OFFSET IN FRAME = {}".format(elapsed_off_percentage))
							print("Label For Frame = {}".format(s_or_ns))
							print()

							# # Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
							# # whilst performing the computations.
							# fig = plt.figure(figsize=(10,5))
							# plt.plot(time_window, signal_window)
							# plt.axvline(float(fall_in_on['start']), color='red', label='Spindle Onset')
							# plt.axvspan(float(fall_in_on['start']), max(time_window), color='red', alpha=0.15)
							# plt.legend()
							# plt.show()

							# Ready to compute QPS parameters and features:
							feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

							# Append feature_df onto final_df:
							if (len(final_df) == 0):
								# i.e. We starting fresh.
								for col in feature_df.columns:
									final_df[col] = feature_df[col]

							else:
								# Need a temporary dataframe:
								tmp = pd.DataFrame()
								for col in feature_df.columns:
									tmp[col] = feature_df[col]

								final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

							annotation_finish = 0

					else:

						# This one is a bit of a paradox. We cannot be waiting to finish yet 
						# barely having walked past the onset. This conditional will NEVER be executed
						s_or_ns = 0

						# Ready to compute QPS parameters and features:
						feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

						# Append feature_df onto final_df:
						if (len(final_df) == 0):
							# i.e. We starting fresh.
							for col in feature_df.columns:
								final_df[col] = feature_df[col]

						else:
							# Need a temporary dataframe:
							tmp = pd.DataFrame()
							for col in feature_df.columns:
								tmp[col] = feature_df[col]

							final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

						annotation_finish = 0;

				else:

					# So NO onset. That's fine. But if we DO see an offset marker still in the frame...
					if (num_offset == 1):

						# Need to check the offset marker:
						elapsed_off = max(time_window) - fall_in_off['end']
						elapsed_off_percentage = float(elapsed_off / Tw * 100)

						# Now, we need to check if the offset is <= 50%. We want it as low as possible
						# as this means the frame is well WITHIN the spindle and has not passed it considerably
						if (elapsed_off_percentage <= 50.0):

							# This means we are in the safe zone to classify the frame as having
							# captured a spindle. Update the label flag:
							s_or_ns = 1

							print("A tentative spindle occurs in this window! Spindle used to compute features:")
							print("NO ONSET IN FRAME")
							print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
							print("Label For Frame = {}".format(s_or_ns))
							print()

							# # Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
							# # whilst performing the computations.
							# fig = plt.figure(figsize=(10,5))
							# plt.plot(time_window, signal_window)
							# plt.axvline(float(fall_in_off['end']), color='blue', label='Spindle End')
							# plt.axvspan(min(time_window), float(fall_in_off['end']), color='red', alpha=0.15)
							# plt.legend()
							# plt.show()

							# Ready to compute QPS parameters and features:
							feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

							# Append feature_df onto final_df:
							if (len(final_df) == 0):
								# i.e. We starting fresh.
								for col in feature_df.columns:
									final_df[col] = feature_df[col]

							else:
								# Need a temporary dataframe:
								tmp = pd.DataFrame()
								for col in feature_df.columns:
									tmp[col] = feature_df[col]

								final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

							# We have not yet passed the offset (> 50%). Hence, we have yet to walk
							# past the annotations. Toggle the annotation_finish flag to 0
							annotation_finish = 0

						else:

							# This means we most likely walked too far over the offset (greater than 50% of the frame length)
							# Furthermore, there's no onset makrer in the frame. THIS makes sense.
							# We can classify the captured frame officially as being non-spindle. 
							# And at this point, we have walked sufficiently past the offset marker such that 
							# we are NO longer in the frame of a spindle.
							s_or_ns = 0

							print("NO ONSET IN FRAME")
							print("Percentage Time Frame Elapsed After OFFSET = {}".format(elapsed_off_percentage))
							print("Label For Frame = {}".format(s_or_ns))
							print()

							# Ready to compute QPS parameters and features:
							feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

							# Append feature_df onto final_df:
							if (len(final_df) == 0):
								# i.e. We starting fresh.
								for col in feature_df.columns:
									final_df[col] = feature_df[col]

							else:
								# Need a temporary dataframe:
								tmp = pd.DataFrame()
								for col in feature_df.columns:
									tmp[col] = feature_df[col]

								final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

							annotation_finish = 1			

					else:

						# This means NEITHER onset or offset are in the frame YET we still have yet
						# to completely pass an event. Classify IMMEDIATELY as a spindle
						s_or_ns = 1;

						print("A tentative spindle occurs in this window! Spindle used to compute features:")
						print("NO ONSET IN FRAME")
						print("NO OFFSET IN FRAME.")
						print("Label For Frame = {}".format(s_or_ns))
						print()

						# # Make a plot of the spindle. Jupyter's "QTConsole" should be able to generate a figure
						# # whilst performing the computations.
						# fig = plt.figure(figsize=(10,5))
						# plt.plot(time_window, signal_window)
						# plt.show()

						# Ready to compute QPS parameters and features:
						feature_df = compute_features(time_window, signal_window, sampling_rate, s_or_ns)

						# Append feature_df onto final_df:
						if (len(final_df) == 0):
							# i.e. We starting fresh.
							for col in feature_df.columns:
								final_df[col] = feature_df[col]

						else:
							# Need a temporary dataframe:
							tmp = pd.DataFrame()
							for col in feature_df.columns:
								tmp[col] = feature_df[col]

							final_df = pd.concat([final_df, tmp], axis=0).reset_index(drop=True)

						# We have still yet to walk completely past the annotated event. 
						annotation_finish = 0


	return final_df

#######################################################
# Functions: butter_bandpass + butter_bandpass_filter #
#######################################################

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Perform zero-phase filtering on the data.
    y = filtfilt(b, a, data)
    return y

#################################
# Function: auto_spindle_detect #
#################################

def auto_spindle_detect(time, signal, sampling_rate):
	
	"""
	Function: 'auto_spindle_detect':

	Parameters:
		- signal: Signal to be used to detect 'tentative' spindles. The function uses a thresholding algorithm to determine a set of onset and offset markers
				  for these tentative spindles. The purpose of these markers is NOT to classify the spindles but for the purposes of the NLLS regression parameter
				  initialisation in the 'compute_features' function. 
	
	Returns:
		- auto_markers: A pandas dataframe containing the onset and offset markers for the tentative spindles detected by the thresholding algorithm. The dataframe
						consists of the onset (start time), the duration of the tentative spindle as well as the offset (end time) of the spindle. 
	"""

	################################################################################
	# Step 1: Bandpass the signal in the 11-16 Hz range using zero-phase filtering #
	################################################################################

	signal_filtered = butter_bandpass_filter(signal, 11, 16, sampling_rate, order=5)

	###################################################################
	# Step 2: Compute the sigma envelope of the signal being analysed #
	###################################################################

	# Generate a smoothing filter with a 200 ms window size:
	smoothing_size = 0.2
	smoothing = np.ones(1, np.floor(smoothing_size * sampling_rate))

	# Compute the Hilbert transform of the signal:
	sig_filt_env = abs(hilbert(signal_filtered))

	# Smooth the Sigma envelope via convolution between the moving average window and the envelope itself:
	sig_filt_env_smooth = np.convolve(sig_filt_env, smoothing)

	###########################################################################################################
	# Step 3: Generate the lower and upper threshold values that will be used for tentative spindle detection #
	###########################################################################################################

	# Compute the upper and lower threshold for spindle detection:
	upper_thresh = 6 * np.median(sig_filt_env_smooth)
	lower_thresh = 2 * np.median(sig_filt_env_smooth)

	################################################################################################################################
	# Step 4: Decision algorithm based on signal meeting BOTH lower and upper thresholds. Create boolean arrays for each threshold #
	################################################################################################################################

	lower_cond = (sig_filt_env_smooth == lower_thresh)
	upper_cond = (sig_filt_env_smooth == upper_thresh)

	#########################################################################################
	# Step 5: Determine the indices where the threshold has been met by the sigma envelope. #
	# Tolerance = 0.05 seconds = 50 ms. Plug the indices into the time-array to determine   #	
	# the times corresponding to the detected arrays:										#
	#########################################################################################

	lower_cond_met = np.where(np.isclose(sig_filt_env_smooth, lower_thresh, atol = 5e-01))
	time_lower = time[lower_cond_met]
	upper_cond_met = np.where(np.isclose(sig_filt_env_smooth, upper_thresh, atol = 5e-01))
	time_upper = time[upper_cond_met]

	# A potential spindle is met when the time-interval detected by the UPPER threshold
	# is bound between a time-interval detected by the LOWER threshold.
	# Potential Spindle Times:
	potential_spindles = [];
	for i in range(len(time_lower) - 1):
	    delta_min = time_lower[i];
	    delta_max = time_lower[i+1];
	    for instance in time_upper:
	        if ((instance > delta_min) & (instance < delta_max)):
	            potential_spindles.append(delta_min)
	            potential_spindles.append(delta_max)

	# However, if the time-span is too short or too long, then we shouldn't count these as spindles. Perhaps we can say
	# that is the time-span of the potential spindles MUST be >= 500 ms (0.5s) and <= to 2000 ms (2s). This is consistent
	# with the AASM definition.
	final_spindles = [];
	for i in range(len(potential_spindles) - 1):
	    # Compute the time-differential:
	    time_diff = potential_spindles[i+1] - potential_spindles[i]
	    # Check if time-diff fits the AASM spindle length criterion:
	    if ((time_diff >= 0.5) & (time_diff <= 2.0)):
	        final_spindles.append(potential_spindles[i])
	        final_spindles.append(potential_spindles[i+1])

	# Convert the final_spindles array into a NumPy array if in need of any processing. We also want the unique instances.
	# That is, no duplicate time samples in the array:
	final_spindles = np.unique(np.array(final_spindles))

	# Separate the instances in the 'final_spindles' array into an array for onset and offset markers
	# Every 'off' array element MUST be an onset while every even array element MUST be an offset.
	onsets = [];
	offsets = [];
	for i in range(len(final_spindles)):
	    if (i % 2 == 0):
	        onsets.append(final_spindles[i])
	    else:
	        offsets.append(final_spindles[i])

	# Compute the duration between the onset markers and their corresponding offsets:
	durations = [];
	for on, off in zip(onsets, offsets):
	    dur = off - on
	    durations.append(dur)

	# Generate a dataframe with Column1 = onsets, Column2 = durations and Column3 = offsets.
	start = pd.Series(onsets)
	duration = pd.Series(durations)
	end = pd.Series(offsets)
	frame = {'start':start, 'duration':duration, 'end':end}
	auto_annotations = pd.DataFrame(frame)

	# Return the annotations created by the automatic threshold detection algorithm #
	return auto_annotations

#######################
# Function: bandpower #
#######################

def bandpower(signal, sampling_rate, freq_range, frame_size, frame_stride, relative=False):

	# Get the lower and upper cutoff frequency as individual variables.
	band = np.asarray(freq_range)
	low, high = band

	# Variable to store the maximum band power within the particular band:
	max_bp = 0

	# Initialise parameters for a moving window. The window scans through the signal
	# computes the PSD in that frame and then moves along based on a stride.
	# The length of this frame is relative to the signal length. For 1.0 second duration signal
	# We should have a frame size half of the size (e.g. 0.5 second). The frame moves along the signal 
	# with overlap perhaps equal to the 1/4 of the frame size (e.g. 0.25 s).

	# Convert the frame size to the equivalent length in samples:
	frame_size_samples = int(frame_size * sampling_rate)

	# Convert the frame stride to the equivalent stride length in samples:
	frame_stride_samples = int(frame_stride * sampling_rate)

	# Let 'curr_frame' be the current frame number:
	curr_frame = 0

	# Compute the upper bound for the current frame. This is based on the length of the signal (which is going to be like 1.0 seconds)
	curr_frame_limit = int( np.ceil((len(signal) - frame_size_samples) / frame_stride_samples) )

	# Create a for-loop that extracts a frame, computes the PSD and stores the bandpower (bp) in max_bp is the current
	# computed bandpower is greater than that of the last frame captured.
	for curr_frame in range(curr_frame_limit):

		# Get the lower and upper limits of the frame being extracted:
		frame_min = 0 + (curr_frame * frame_stride_samples)
		frame_max = frame_size_samples + (curr_frame * frame_stride_samples)

		# Assign 'data' as the frame extracted from the signal
		data = signal[frame_min:frame_max]

		# Computing the PSD. Frequency-axis goes up to the Nyquist frequency which is Fs/2.
		psd, freqs = psd_array_multitaper(data, sampling_rate, adaptive=True, normalization='full', verbose=0)

		# The frequency resolution (df) necessary for integral calculation via simpson's rule
		df = freqs[1] - freqs[0]

		# Find index of band in frequency vector
		idx_band = np.logical_and(freqs >= low, freqs <= high)

		# Integral approximation of the spectrum using parabola (Simpson's rule)
		bp = simps(psd[idx_band], dx=df)

		if relative == True:
		    bp = bp / simps(psd, dx=df)

		if bp >= max_bp:
			max_bp = bp;

	# Return the final bandpower value (Relative value recommended)
	return max_bp

##############################
# Function: compute_features #
##############################

def compute_features(time_window, signal_window, sampling_rate, manual_label):

	"""
	Function: 'compute_features'

	Function accepts a frame from 'manual_class_and_qps' and computes the QPS parameters from the spindle via NLLS.
	The parameter values are initialised from values known a posteriori (from other papers + previous analysis of the MASS
	spindles). The function also computers other features such as the residual and QPS energy and more. These features are
	defined in more depth in the final report.

	Parameters:
		- time_window: The time-vector corresponding to the raw signal.
		- signal_window: A raw portion of an epoch captured in the manual_class_and_qps function
		- sampling_rate: Sampling rate used to record the EEG signal.
		- manual_label: The label associated with the annotations generated from the automatic detection algorithm.

	Returns:
		- feature_df: Pandas dataframe containing the features computed from the raw signal passed in as input.

	"""

	# Set up the time-vector for the NLLS fitting:
	tmp = len(time_window)
	sampling_rate = sampling_rate
	t = np.arange(-tmp/(2*sampling_rate), (tmp-1)/(2*sampling_rate), 1/sampling_rate)

	# Set up handler function to be used to compute and minimise the residual via 'lmfit'
	def residual(params, t, data):
		a, b, c = params['a'], params['b'], params['c']
		d, e, f = params['d'], params['e'], params['f']

		# The overall envelope MUST be positive. If a negative value is present, set to 0.
		envelope = np.exp(a + b*t + c*t**2)
		
		for i in range(len(envelope)):
			if envelope[i] < 0:
				envelope[i] = 0

		carrier = np.cos(d + e*t + f*t**2)

		model = envelope * carrier 
		chi = (data - model)
		return chi

	# Set up initialised values for the QPS parameters if manual_label == 1
	# a, b and c from previous analysis of MASS dataset
	# d, e and f from Kulkarni et. al.

	#####################################################################################################################################
	# Here, we use the SDT ratio in order to perform the NLLS parameter initialisation. The greatest separation between the spindles
	# and non-spindles is when we initialise spindles as the mean values (known a priori) and the non-spindles to 0. In this sense
	# Analysis of MASS Patient #1 showed that the mean SDT ratio was found to be 0.368227 (0.263520) where the number in the brackets
	# is the standard deviation of the SDT ratio. We use this range to perform the initialisation.
	#####################################################################################################################################

	###################################################################################
	# Calculate the spindle-to-(delta+theta) Ratio (SDT Ratio) with Multitaper method #
	###################################################################################

	psd_frame_size = 0.5 		# 0.5-second
	psd_frame_stride = 0.25 	# 0.25 of a second

	spindle_bp = bandpower(signal_window, sampling_rate, [11,16], psd_frame_size, psd_frame_stride, relative=True)
	delta_bp = bandpower(signal_window, sampling_rate, [0.5, 4], psd_frame_size, psd_frame_stride, relative=True)
	theta_bp = bandpower(signal_window, sampling_rate, [4,7], psd_frame_size, psd_frame_stride, relative=True)

	# Compute the SDT Ratio:
	sdt_ratio = spindle_bp / (delta_bp + theta_bp)

	# Use a conditional to initialise the parameters:
	if ((sdt_ratio >= 0.368227 - 0.263520) & (sdt_ratio <= 0.368227 + 0.263520)):

		a = 0.82	
		b = 1.05
		c = -10
		d = 0
		e = 84.5
		f = -0.9

	else:

		a = 0.0
		b = 0.0
		c = 0.0
		d = 0.0
		e = 0.0
		f = 0.0

	# Plug in the initialised parameters into lmfit's 'Parameters' function:
	params = Parameters()
	params.add('a', value=a, min=-50.0, max=10)
	params.add('b', value=b)
	params.add('c', value=c)
	params.add('d', value=d)
	params.add('e', value=e)
	params.add('f', value=f)

	# Before minimisation, need to ensure all NaN values have been converted to some non-NaN value.
	t = np.array(pd.Series(t).interpolate(method='linear').fillna(0))
	signal_window = np.array(pd.Series(signal_window).interpolate(method='linear').fillna(0))

	# Perform a bandpass filtering operation on the signal_window:
	signal_window_filtered = butter_bandpass_filter(signal_window, 11, 16, sampling_rate, order=5)

	# Perform the minimisation via NLLS. 
	out = minimize(residual, params, args=(t, signal_window_filtered))

	# We now need to gather all the parameter values from the 'out' return from the minimize function
	dict_params = {'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0}
	list_params = ['a','b','c','d','e','f']
	for param in list_params:
	    dict_params[param] = out.params[param].value
	  
	# Generate the QPS model from the particular signal captured by the window:  
	a, b, c = dict_params["a"], dict_params["b"], dict_params["c"]
	d, e, f = dict_params["d"], dict_params["e"], dict_params["f"]
	qps = np.exp(a + b*t + c*t**2) * np.cos(d + e*t + f*t**2)

	#######################################################
	# Compute the residual from the signal_window and qps #
	#######################################################

	raw = signal_window_filtered
	residual = raw - qps

	# Converting the dictionary of QPS parameter values into a pandas dataframe. The indices are not crucial at this point
	param_df = pd.DataFrame(dict_params, index=[0,])

	#######################
	# Energy Calculations #
	#######################

	# Compute Energy of QPS 
	qps_energy = np.sum(np.square(qps))
	param_df["qps_energy"] = qps_energy

	# Compute Energy Of REAL Spindle
	real_energy = np.sum(np.square(signal_window))
	param_df["real_energy"] = real_energy

	# Compute Energy Of Residual 
	residual_energy = np.sum(np.square(residual))
	param_df["residual_energy"] = residual_energy

	# QPS-Spindle Energy Ratio (QSER)
	param_df['qser'] = qps_energy / real_energy

	# Residual-to-Spindle Energy Ratio (RSER) 
	param_df['rser'] = residual_energy / real_energy

	# QPS Spindle Energy Error Percentage (Can Be Used For Boxplot)
	param_df['energy_error_percent'] = (np.abs(qps_energy - real_energy) / real_energy) * 100.0

	##########################
	# Frequency Calculations #
	##########################

	# Compute the peak frequency of the band-pass filtered signal:
	hann_window = hann(len(signal_window_filtered))						# Creating a hanning window to multiply with the captured signal.
	N = len(signal_window_filtered)										# Compute length of signal
	freqx = fftfreq(N, 1/sampling_rate)									# Calculate frequency bins (x-axis)
	freqy = (2/N) * abs(fft(signal_window_filtered * hann_window))		# Compute the magnitude spectrum of the real spindle. Need a window to ensure no spectral leakage
	centre_freq_index = np.where(freqy == max(freqy))					# Determine the array index (in freqy) where the centre (max) frequency occurs
	real_freq = abs(freqx[centre_freq_index[0][0]])						# Obtain the real frequency from the freqx array
	param_df["real_freq_hz"] = real_freq								# Assign as another column of a dataframe

	# Compute the peak frequency based off the generated QPS (parameter 'e'):
	qps_freq = e / (2*np.pi)
	param_df['qps_freq_hz'] = qps_freq

	# Compute the percentage frequency error between the actual spindle and the QPS spindle (Can Be Used For Boxplot)
	param_df['freq_error_percent'] = (np.abs(qps_freq - real_freq) / real_freq) * 100.0

	#######################################################
	# Correlation Coefficient Between Raw and QPS Spindle #
	#######################################################

	param_df['raw_qps_corrcoeff'] = np.corrcoef(pd.Series(qps),pd.Series(raw))[0,1]
	R = np.corrcoef(pd.Series(qps),pd.Series(raw))[0,1]

	#######################################################################
	# Add the SDT ratio as a feature in the dataframe. May or may not use #
	#######################################################################

	param_df['sdt_ratio'] = sdt_ratio

	###########################################################
	# Manually classify the frame based on the expert scorer. #
	###########################################################

	param_df['label'] = manual_label

	##########################################################
	# Return the dataframe containing all computed features: #
	##########################################################

	feature_df = param_df.copy()

	return feature_df

##################
# Function: Main #
##################

def main():

	# Request user for the absolute filepath containing the PSG files:
	absolute_filepath = input("Enter absolute filepath containing the PSG recordings and annotations: ")

	# Request user for the filename of the PSG recording
	psg_rec = input("Enter filename (including file extention) of the PSG recording (preferably in EDF++): ")

	# Request user for the sleep stage annotations:
	sleep_stage_annot = input("Enter filename (including file extension) for the sleep stage annotations (preferably in EDF++): ")

	# Ask if user wants to use a single scorer annotation or BOTH scorer annotations (if there are two):
	scorer_request = int(input("Use SINGLE scorer annotation (Enter '1') OR BOTH scorer annotations (Enter '2'): "))
	
	if (scorer_request == 1):

		scorer_annot_1 = input("Enter filename (in EDF++) for Scorer 1's annotation: ")
		scorer_annot_2 = None

	elif (scorer_request == 2):

		scorer_annot_1 = input("Enter filename (in EDF++) for Scorer 1's annotation: ")
		scorer_annot_2 = input("Enter filename (in EDF++) for Scorer 2's annotation: ")

	# Pass all parameters into 'read_eeg_and_annot' function. Get all necessary returns from the function
	(raw, epochs_df, scorer_intersection, sampling_rate) = read_eeg_and_annot(absolute_filepath, psg_rec, sleep_stage_annot, scorer_annot_1, scorer_annot_2)

	# Isolate desired epochs. Request from user the desired sleep stage and PSG channel name:
	sleep_stage = input("Enter sleep stage to be extracted from PSG: ")
	ch_name = input("Enter the PSG channel name to be extracted: ")
	(eeg_epochs, epoch_list) = epochs_isolate(epochs_df, sleep_stage, ch_name)

	print() # New line

	print("DataFrame containing all the epochs for the channel '{}' \n".format(ch_name))
 
	# Print out the EEG epochs for verification on the terminal.
	print(eeg_epochs)
	print() # New line

	# Print out the epoch number of the VERY last epoch.
	print("Last epoch in the time-series: Epoch #{}".format(epoch_list[-1]))

	# Once the sleep stage and its epochs have been filtered, we can begin the classification and QPS parameter computation:
	# Request user for window length and stride to be used during the computation:
	print("The next stage uses a sliding window across the 20.0 second epochs to classify the captured signal as a spindle or not. \n")
	window_length = float(input("Enter a window length (in seconds): "))
	window_stride = float(input("Enter the window stride (in seconds). For overlap, set stride less than window length: "))

	# Request user for the final filename containing the features and labels (in .csv format):
	csv_filename = input("Enter a filename for the .csv file containing the features and labels: ")
	final_csv_path = absolute_filepath + "\\" + csv_filename
	final_df = manual_class_and_qps(eeg_epochs, epoch_list, scorer_intersection, ch_name, window_length, window_stride, sampling_rate)

	# Save the DataFrame as a .csv file in the same directory containing all the raw files.
	final_df.to_csv(final_csv_path, index = False)

################
# Execute main #
################

main()






























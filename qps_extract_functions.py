# Import main libraries for data processing:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import signal processing functions from Scipy
from scipy.signal import butter, lfilter, sosfilt, filtfilt

# Set up seaborn if plotting is performed
sns.set()

# Import libraries for reading EDF files (if necessary):
import mne

# import libraries for non-linear regression and parameter extraction
from lmfit import Parameters, minimize

# For generating a QPS waveform:
from wavelet_generator import qps_time_sig

#############################
# Function: 'read_eeg_file' #
#############################

def read_eeg_and_annot(absolute_filepath, eeg_excerpt, visual_annot, sampling_rate):
	"""
	Function takes in the filenames of the EEG excerpt (as a .txt file) and the
	associated annotations for micro events in the EEG excerpt. Function returns
	a 1D NumPy array of the EEG excerpt and a 2D NumPy array comprising of the
	start and end times of microevents in terms of samples. More on parameters below:

	Parameters:
	- argv[0]: Absolute filepath for the folder containing all EEG excerpts and annotations
	- argv[1]: Filename for the EEG excerpt to be read and processed
	- argv[2]: Filename for the visual annotations of microevents in the corresponding excerpt
	- argv[3]: The sampling rate used for the EEG recording.

	Returns:
	- 1. 'eeg': The processed EEG recording as a 1D NumPy array
	- 2. 'microevents': The processed visual annotations as a 2D array where:
			- Column 1 indicates the start time of an event (in samples)
			- Column 2 indicates the end time of an event (in samples)
	"""


	#################################
	# Reading the required EEG file #
	####################################################################################
	# To date, this is performed using the .txt files provided by the DREAMS database ##
	# The EDF files, at the moment, cannot be loaded with mne. 						  ##
	####################################################################################

	# Assign variable with filepath to the dreams folder in local drive.
	filepath = absolute_filepath

	# Request for EEG filename from user:
	eeg_file = eeg_excerpt

	# Request for visual Spindle annotations (by expert):
	spindle_marks = visual_annot

	# Sampling rate (fs) used for recording the EEG:
	sampling_rate = sampling_rate
	
	#############################################
	# Processing the EEG and visual annotations #
	#############################################

	# Convert EEG text file into a 1D Numpy Array and get the length of the EEG recording (in samples)
	total_filepath = filepath + "\\" + eeg_file
	raw = pd.read_csv(total_filepath, sep='\n')
	raw = np.array(raw).flatten().T

	# Convert visual annotations to pandas then 2D numpy array
	temp = filepath + "\\" + spindle_marks
	expert_spindles = pd.read_csv(temp, sep='\t').reset_index()
	expert_spindles.columns = ['time', 'duration']
	microevents_start = (np.array(expert_spindles['time']) * sampling_rate).astype(int)
	microevents_duration = (np.array(expert_spindles['duration']) * sampling_rate).astype(int)
	microevents_end = np.sum([microevents_start, microevents_duration], axis=0)
	d = {'start':microevents_start, 'end':microevents_end}
	spindle_events = pd.DataFrame(data=d)
	spindle_events = spindle_events.values
	num_of_events = len(spindle_events)


	return (raw, spindle_events, num_of_events)

#######################
# Function: 'eegplot' #
#######################

def eegplot(raw, spindle_events, sampling_rate):

	"""
	Function takes in the processed raw EEG excerpt and the microevents returned by 'read_eeg_and_annot'
	and produced a plot of the EEG with the microevents as an overlay on the EEG plot.

	Parameters:
	- argv[0]: 'raw': The raw EEG as a 1D numpy array returned from read_eeg_and_annot
	- argv[1]: 'spindle_events': The processed microevent start and end times as a 2D NumPy array returned by 'read_eeg_and_annot'
	- argv[2]: 'sampling rate': The sampling rate used when recording the EEG excerpt

	Returns:
	- No returns

	"""

	# Get the length of the EEG excerpt in terms of samples. Create a time axis by dividing with the sampling rate
	total_samples = len(raw)
	max_time = total_samples / sampling_rate
	t = np.arange(0, max_time, 1/sampling_rate)

	# Instantiate a matplotlib figure:
	fig = plt.figure()

	# Plot the raw signal against the time:
	plt.xlabel("Time (Seconds)")
	plt.ylabel("Amplitude ($\mu V$)")
	plt.title("30 minute EEG recording")
	plt.plot(t, raw)

	# Plotting microevents as an overlay on top of EEG plot:
	for event in np.arange(len(spindle_events)):
		plt.axvspan(spindle_events[event][0]/sampling_rate, spindle_events[event][1]/sampling_rate, alpha=0.35, color='red')

	# Show the final plot:
	plt.show()

###############################
# Function: 'butter_bandpass' #
###############################

def butter_bandpass(lowcut, highcut, fs, order = 5):
	"""
	Function to create a prototype nth order butterworth bandpass filter. 
	Returns the numberator (b) and denominator (a) of the transfer function 
	for the bandpass filter.

	Parameters:
	- argv[0]: 'lowcut': The lower cutoff frequency for the bandpass filter
	- argv[1]: 'highcut': The higher cutoff frequency for the bandpass filter
	- argv[2]: 'fs': The sampling frequency to be used for the filter. This should 
					 equal to the sampling frequency used for the EEG recording
	- argv[3]: 'order': Set default to '5', the order determines the rate of attenuation
						before and after the cutoff frequencies (i.e. the passband of the filter.
						NOTE: The higher the order, the greater the delay introduced during the filtering.

	Returns:
	- 1. 'b': Numerator of the transfer function
	- 2. 'a': Denominator of the transfer function for the bandpass filter
	"""
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b,a = butter(order, [low, high], btype='band', output='ba')
	return b,a

######################################
# Function: 'butter_bandpass_filter' #
######################################

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	"""
	Function takes in the EEG data to be filtered and applies a bandpass filter
	based on the lower and higher cutoff frequencies passed as input parameters.
	The sampling frequency must also be provided as an input. Order need not be specified.

	Parameters: 
	- argv[0]: 'data' - The EEG data to be filtered at whatever length
	- argv[1]: 'lowcut' - The lower cutoff frequency (in Hz)
	- argv[2]: 'highcut' - The higher cutoff frequency (in Hz)
	- argv[3]: 'fs': Sampling rate equal to that used when recording the EEG excerpt in the first place.
	- argv[4]: 'order': Set default to 5. Can be changed if needed.

	Returns:
	- y: The filtered EEG data after bandpass filtering
	"""
	b,a = butter_bandpass(lowcut, highcut, fs, order=order)
	y = filtfilt(b, a, data, padlen=25)
	return y

###############################
# Function: 'butter_bandstop' #
###############################

def butter_bandstop(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b,a = butter(order, [low, high], btype='bandstop', output='ba')
	return b,a

######################################
# Function: 'butter_bandstop_filter' #
######################################

def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
	b,a = butter_bandstop(lowcut, highcut, fs, order=5)
	y = filtfilt(b, a, data, padlen=25)
	return y

################################
# Function: 'extract_spindles' #
################################

def extract_spindle(raw, spindle_events, event_number, fl, fh, sampling_rate):
	"""
	Function extracts a SINGLE spindle from a chosen event number. The function uses the bandpass
	filter functions in order to isolate frequencies within the general AASM range of 11-16 Hz 
	or in the typical spindle range of 12-14 Hz. 
	It is intended that the return from 'extract_spindle' be used to extract QPS parameter
	values that are then collected in a csv file in the end.

	Parameters:
	- argv[0]: 'raw': The raw EEG data to be processed to pull a microevent from
	- argv[1]: 'spindle_events': The 2D NumPy array containing the start and end of microevents in the EEG recording.
								 To be used in a for-loop as well.
	- argv[2]: 'event_number': The current event number being accessed via a for-loop (likely through a for-loop index)

	Returns: 
	- marked_spindle_filtered.
	"""
	# Print out event number:
	print("Event Number = {}".format(event_number))

	event = event_number

	# Pull out the spindle at the relevant event number:
	marked_spindle = raw[(spindle_events[event][0]) : (spindle_events[event][1])]

	# Compute the duration of the spindle
	duration = (spindle_events[event][0]) - (spindle_events[event][1]) / sampling_rate

	# Perform the bandpass filtering on the marked-spindle to produced the filtered version within the specified passband
	marked_spindle_filtered = marked_spindle_filtered = butter_bandpass_filter(marked_spindle, fl, fh, 100, order=3)

	# Return the filtered marked spindle:
	return marked_spindle_filtered

####################################
# Function: 'extract_eeg_baseline' #
####################################

def extract_eeg_baseline(raw, spindle_events, event, fl, fh, sampling_rate):
	# Print out event number:
	print("Event Number = {}".format(event))

	# Pull out the spindle at the relevant event number:
	marked_spindle = raw[(spindle_events[event][0]) : (spindle_events[event][1])]

	# Compute the duration of the spindle
	duration = (spindle_events[event][0]) - (spindle_events[event][1]) / sampling_rate

	# Perform the bandpass filtering on the marked-spindle to produced the filtered version within the specified passband
	marked_spindle_filtered = marked_spindle_filtered = butter_bandstop_filter(marked_spindle, fl, fh, sampling_rate, order=3)

	# Return the filtered marked spindle:
	return marked_spindle_filtered

#######################
# Function: 'fit_qps' #
#######################

def fit_qps(marked_spindle_filtered, sampling_rate, qps_param_filename):

	"""
	Function fits the QPS model for sleep spindles and extracts the 6 parameter values associated with the QPS model.

	"""

	# Generate the QPS model using the same time interval as the marked_spindle_filtered
	tmp = len(marked_spindle_filtered)
	t_qps = np.arange(-tmp/(2*sampling_rate), (tmp -1)/(2*sampling_rate), 1/sampling_rate)
	
	# Perform a non-linear regression to fit the QPS model to the raw spindle
	def residual(params, t, data):
		a, b, c = params['a'], params['b'], params['c']
		d, e, f = params['d'], params['e'], params['f']
		model = np.exp(a + b*t + c*t**2) * np.cos(d + e*t + f*t**2)
		chi = (data - model)
		return chi

	# Set up an empty set of parameters:
	# Option 1: If a best fit is desired, set parameter values to the mean values for an average sleep spindle. Something like:
	#			a=2.5, b=0, c=-20, d=35, e=81.681, f=13.6
	# Option 2: For non-sleep spindles, set all parameters to 0. This prevents the NLLS fitting to work properly.

	# Set up an empty set of parameters:
	# a=2.5, b=0, c=-20, d=35, e=81.681, f=13.6
	params = Parameters()
	params.add('a', value=2.5)
	params.add('b', value=0)
	params.add('c', value=-20)
	params.add('d', value=35)
	params.add('e', value=81.681)
	params.add('f', value=13.6)

	# params = Parameters()
	# params.add('a', value=0)
	# params.add('b', value=0)
	# params.add('c', value=0)
	# params.add('d', value=0)
	# params.add('e', value=0)
	# params.add('f', value=0)
	    
	out = minimize(residual, params, args=(t_qps, marked_spindle_filtered))

	# We now need to gather all the parameter values from the 'out' return from the minimize function
	dict_params = {'a':0, 'b':0, 'c':0, 'd':0, 'e':0, 'f':0}
	list_params = ['a','b','c','d','e','f']

	for param in list_params:
	    dict_params[param] = out.params[param].value

	# Converting the dictionary of QPS parameter values into a pandas dataframe. The indices are not crucial at this point
	param_df = pd.DataFrame(dict_params, index=[0,])

	# Append to a new '.csv' file the values of the QPS parameters from the param_df dataframe:
	param_df.to_csv(qps_param_filename, mode='a', header=False, index=False)

	# Return most recent parameter set IF using the function as a one-off (perhaps in a Jupyter Notebook)
	return param_df

############################
# Function: compute_energy #
############################

def compute_qps_energy(qps_row, sampling_rate, duration):
	"""
	compute_qps_energy: Function accepts a set of QPS parameter values from a dataset (a single row) where each parameter corresponds 
	to a column of the dataset. The function then generates an amplitude modulated QPS descriptor and computes that energy of the
	resultant QPS. Recall that the energy of a finite signal is the inner product of the signal with itself or the integral of the signal-squared.

	Parameters:
		- argv[0]: qps_row: Row from a pandas dataframe. Treat as dataframe itself comprising of only ONE row. The row contains entries for 
							six columns where each column represents a particular QPS parameter. The last column of the dataframe should
							be the documented duration of the visually detected spindle (to be passed as the third parameter)
		- argv[1]: sampling_rate: Sampling rate used to record the original EEG spindle. This, of course, depends on the excerpt (subject)
								  recorded in the first place. Each dataset will have to be treated separately before any dataset concatenation
								  can occur.
		- argv[2]: duration: This is the duration (in seconds) of the visually detected spindle. The duration will then be used to create a time
							 interval for the QPS spindle with respect to the parameters.

	Returns:
		- qps_energy: This is the energy of the QPS spindle based off the parameter values, duration and sampling rate. The energy is used
					  can be used as a possible feature in a machine learning algorithm.

	"""
	# Assign variables for the parameters from each column of qps_row:
	a, b, c = qps_row["a"], qps_row["b"], qps_row["c"]
	d, e, f = qps_row["d"], qps_row["e"], qps_row["f"]

	# Create a time-interval for the QPS based on the duration of the visually detected spindle.
	half_duration = duration / 2
	period = 1 / sampling_rate
	t = np.arange(-half_duration, half_duration, period)

	# Generate the QPS based on the parameters:
	qps = np.exp(a + b*t + c*t**2) * np.cos(d + e*t + f*t**2)

	# Compute the energy of the QPS by calculating the sum of the discrete QPS values squared.
	qps_energy = np.sum(np.square(qps))

	# The resultant energy value will be in (microvolt)^2 units. Return the energy value:
	return qps_energy



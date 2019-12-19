import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

sns.set()

def per_subject_boxplot(dict_1, dict_2, spindle_subject_key1, spindle_subject_key2, column_names, subject_number):

	"""
	Function Name: 'per_subject_boxplot'

	Purpose: Function takes in TWO dictionarys containing more than one dataframe (value), accessible via keys.
			 The function takes in a particular dict key to access a specific dataframe. Since the dataframes
			 have columns, a list of relevant column names is also passed into the function. The number of rows
			 and columns are determined by the number of columns, of the dataframe, to be plotted. For instance,
			 if there are 6 columns, the boxplots are arranged in a 2*6 arrangement.

			 The function is to be used in a conjunction with an instantiated Dropdown ipywidget. Upon a change
			 in event in the dropdown (i.e. selecting the DREAMS subject number), this function should trigger.
			 In the main function, a conditional block should be executed to select the relevant subject key
			 name by the subject number. The subject number should be passed in as the last argument to ensure
			 the boxplots visualise TWO or ONE box-and-whisker plots.

	Parameters: 
	- argv[0]: dict_1 (a dictionary)
	- argv[1]: dict_2 (a dictionary)
	- argv[2]: spindle_subject_key1 (a string)
	- argv[3]: spindle_subject_key2 (a string)
	- argv[4]: column_names (a list)
	- argv[5]: subject_number

	Returns:
	- No return. Just the boxplot output as a matplotlib function
	"""

	# Step 1: Access the dataframe based on the dictionary key. The 'subject_key_csv_name' is pulled in the main function
	s_df_1 = dict_1[spindle_subject_key1]
	s_df_2 = dict_2[spindle_subject_key2]

	# Step 2: Proceed to create a matplotlib subplots figure. The dimensions are 'r * 3' where r is the number of rows.
	#		  It is expected that the number of columns be some multiple of 3.
	nrows = 2
	ncols = 3
	k = subject_number
	c = 0 # To iterate through the features in the column_names list
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(12,6))

	# Step 3: Use a for-loop to generate the boxplots for the particular DREAMS subject:
	for row in range(nrows):
		for col in range(ncols):

			# Case 1: Subject number is from 1-6 (inclusive). Plot boxplots for BOTH visual scorer 1 and 2
			if (k <= 6):
				ax[row,col].boxplot([s_df_1[column_names[c]], s_df_2[column_names[c]]], showfliers=False)
				ax[row,col].set_title("Distribution for Parameter {}".format(column_names[c]))

				# Print out the mean and standard deviation for each parameter for the subject:
				mean1, std1 = round(s_df_1[column_names[c]].mean(), 2), round(s_df_1[column_names[c]].std(), 2)
				mean2, std2 = round(s_df_2[column_names[c]].mean(), 2), round(s_df_2[column_names[c]].std(), 2)

				print("Parameter '{}' Mu(Sigma): \t  Scorer 1 = {}({}) \t Scorer 2 = {}({})".format(column_names[c], mean1, std1, mean2, std2))

				c = c + 1

			# Case 2: Subject number is 7-8. Plot boxplots for ONLY visual scorer 1
			if (k > 6):
				ax[row,col].boxplot([s_df_1[column_names[c]]], showfliers=False)
				ax[row,col].set_title("Distribution for Parameter {}".format(column_names[c]))

				# Print out the mean and standard deviation for each parameter for the subject:
				mean1, std1 = round(s_df_1[column_names[c]].mean(), 2), round(s_df_1[column_names[c]].std(), 2)

				print("Parameter '{}' Mu(Sigma): \t  Scorer 1 = {}({})".format(column_names[c], mean1, std1))

				c = c + 1

def per_subject_boxplot_engineered(dict_1, dict_2, spindle_subject_key1, spindle_subject_key2, column_names, subject_number):

	"""
	Function Name: 'per_subject_boxplot_engineered'

	Purpose: Same thing, but for the engineered features!

	Parameters: 
	- argv[0]: dict_1 (a dictionary)
	- argv[1]: dict_2 (a dictionary)
	- argv[2]: spindle_subject_key1 (a string)
	- argv[3]: spindle_subject_key2 (a string)
	- argv[4]: column_names (a list)
	- argv[5]: subject_number

	Returns:
	- No return. Just the boxplot output as a matplotlib function
	"""

	# Step 1: Access the dataframe based on the dictionary key. The 'subject_key_csv_name' is pulled in the main function
	s_df_1 = dict_1[spindle_subject_key1]
	s_df_2 = dict_2[spindle_subject_key2]

	# Step 2: Proceed to create a matplotlib subplots figure. The dimensions are 'r * 3' where r is the number of rows.
	#		  It is expected that the number of columns be some multiple of 3.
	nrows = 2 # (The ONLY change is the number of rows)
	ncols = 4
	k = subject_number
	c = 0 # To iterate through the features in the column_names list
	fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(12,6))

	# Step 3: Use a for-loop to generate the boxplots for the particular DREAMS subject:
	for row in range(nrows):
		for col in range(ncols):

			# Case 1: Subject number is from 1-6 (inclusive). Plot boxplots for BOTH visual scorer 1 and 2
			if (k <= 6):
				ax[row,col].boxplot([s_df_1[column_names[c]], s_df_2[column_names[c]]], showfliers=False)
				ax[row,col].set_title("Parameter '{}'".format(column_names[c]))

				# Print out the mean and standard deviation for each parameter for the subject:
				mean1, std1 = round(s_df_1[column_names[c]].mean(), 2), round(s_df_1[column_names[c]].std(), 2)
				mean2, std2 = round(s_df_2[column_names[c]].mean(), 2), round(s_df_2[column_names[c]].std(), 2)

				print("Parameter '{}' Mu(Sigma): \t  Scorer 1 = {}({}) \t Scorer 2 = {}({})".format(column_names[c], mean1, std1, mean2, std2))

				# Iterate to next feature:
				c = c + 1

			# Case 2: Subject number is 7-8. Plot boxplots for ONLY visual scorer 1
			elif (k > 6):
				ax[row,col].boxplot([s_df_1[column_names[c]]], showfliers=False)
				ax[row,col].set_title("Parameter '{}'".format(column_names[c]))

				# Print out the mean and standard deviation for each parameter for the subject:
				mean1, std1 = round(s_df_1[column_names[c]].mean(), 2), round(s_df_1[column_names[c]].std(), 2)

				print("Parameter '{}' Mu(Sigma): \t  Scorer 1 = {}({})".format(column_names[c], mean1, std1))

				c = c + 1

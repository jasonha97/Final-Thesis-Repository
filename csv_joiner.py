################################
# Script: 'csv_joiner'         #
# Date: 7/8/2019               #
# Author: Jason Ha             #
################################################################################################################
# This script aims to be a quick method to join a bunch of separate csv into a single dataframe that can be    #
# exported as a single .csv file (at the users request) 													   #
################################################################################################################

import numpy as np
import pandas as pd

#################
# Main Function #
#################

def main():

	# Initiate the script by loading in the first .csv file (to be appended onto):
	first_file = input("Enter the ABSOLUTE filepath of the first .csv file: ")

	# If the file is dragged-dropped onto the command-line interface, we need to remove quotation marks before loading the file:
	first_file = first_file[1:len(first_file)-1]

	df = pd.read_csv(first_file)

	# Create a flag to indicate whether user wants to continually append .csv files:
	continue_append = 1

	while (continue_append == 1):

		# Ask user if they want to continue appending csv files to the original 'df' DataFrame
		continue_append = int(input("Append a .csv file? Enter '1' for YES. Enter '0' for NO: "))

		# Print new line...
		print()

		if (continue_append == 0):

			# Break from loop and quit program:
			break

		else:

			# Print new line...
			print()

			# Otherwise, we append a fresh .csv file to 'df'. Ask user for the filename:
			next_file = input("Enter the ABSOLUTE filepath of the NEXT .csv file: ")
			next_file = next_file[1:len(next_file)-1]
			tmp = pd.read_csv(next_file)

			# Then, append 'tmp' onto 'df' along the rows:
			df = pd.concat([df, tmp], axis=0)

			# Reset the indices again and drop the automatically generated 'index' column:
			df = df.reset_index(drop=True)

	# If the user breaks from the loop, we need to save the resultant dataframe into a fresh .csv file. Ask user for new filename:
	new_filepath = input("Enter filepath to save new .csv file: ")
	new_filename = input("Enter filename for the new .csv file: ")
	new_file = new_filepath + "\\" + new_filename
	df.to_csv(new_file)

	return 0

##########################
# Call the main function #
##########################

main()



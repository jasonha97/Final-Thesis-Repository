import re
import pandas as pd
import mne

def read_edf_annotations(fname):
    """read_edf_annotations

    Parameters:
    -----------
    fname : str
        Path to file.

    Returns:
    --------
    annot : DataFrame
        The annotations
    """
    with open(fname, 'r', encoding='utf-8',
              errors='ignore') as annotions_file:
        tal_str = annotions_file.read()

    exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
          '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
          '(\x14(?P<description>[^\x00]*))?' + '(?:\x14\x00)'

    annot = [m.groupdict() for m in re.finditer(exp, tal_str)]

    good_annot = pd.DataFrame(annot)
    good_annot = good_annot.query('description != ""').copy()
    good_annot.loc[:, 'duration'] = good_annot['duration'].astype(float)
    good_annot.loc[:, 'onset'] = good_annot['onset'].astype(float)
    return good_annot

def set_the_annotations(hypnogram_filename, raw):

	""" set_the_annotations

	Parameters:
	-----------
	hypnogram_filename : str
		Path to hypnogram file (in EDF format)

	raw : mne.io raw edf file
		Raw PSG recording that has been read initially beforehand

	Returns:
	--------
	No return. Just sets the annotations to the raw PSG.

	"""

	# Function takes in the hypnogram/annotations (in EDF) as input. Returns the annotations as a return
	annot = read_edf_annotations(hypnogram_filename)

	# mne.Annotations takes in the onset and duration of each annotation (e.g. N1, N2, REM etc.) and their description. Return are the annotations
	mne_annot = mne.Annotations(annot.onset, annot.duration, annot.description)

	# The following line allows the annotations to be set on top of the current PSG recording, where 'raw' is the raw PSG file loaded into the worksapce via MNE.
	raw.set_annotations(mne_annot)

	print(mne_annot)
    
def raw_to_df(raw):
    
    events, event_id = mne.events_from_annotations(raw)
    
    epochs = mne.Epochs(raw, events, event_id, tmin=0.0, tmax=15.0, baseline=None)
    
    index, scaling_time, scalings = ['epoch','time'], 1e3, dict(grad=1e13)
    
    df = epochs.to_data_frame(picks=None, scalings=scalings, scaling_time=scaling_time, index=index)
    
    return(epochs, df)
    
    
    
    
    
    
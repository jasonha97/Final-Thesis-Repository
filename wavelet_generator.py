import numpy as np
import pandas as pd

def qps_time_sig(t, a=1,b=1,c=1,d=1,e=1,f=1):
    
	""" qps_time_sig

	Preamble:
	---------
	The QPS wavelet is an anaytic signal that takes the form:

		s(t) = exp(P(t)) * cos(Q(t))

	Where:

		P(t) = a + bt + ct**2
		Q(t) = d + et + ft**2

	Parameters:
	-----------
		a: Some real float value - Varies the amplitude of the signal
		b: Varies the...
		c: Varies the rate of decay of the Gaussian envelope
		d: Phase offset of the carrier component
		e: Frequency of the carrier component
		f: Varies the intra-spindle frequency for sensical values (e.g. f = 14, 23 etc.). Extreme values cause asymmetry to the spindle.
		t_vec: A time vector for the spindle. Imposes a time-interval for the spindle to be generated within.

	Returns:
	--------
		s: The generated QPS wavelet with the associated parameter values given as inputs.

	"""

	s = np.exp(a + b*t + c*t**2) * np.cos(d + e*t + f*t**2)

	return s

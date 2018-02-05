

import numpy as np

#
# get iir filter coefficient
#
# unit: fc is Hz, gain is dB, Q is 2nd order filter of Q
#
# CUT is 2nd order BPF, used for notch filter


FS0=48000

class IIR1:
	def __init__(self,fs0=FS0):
		self.FS=fs0

	def CUT(self,fc,gain,Q):
		a = [0.0] * 3
		b = [0.0] * 3
		a[0]=1.0
		b[0]=1.0
		wc= 2.0 * np.pi * fc /self.FS
		kc= np.power(10.0, (gain /20.0)) -1.0
		x0= 1.0 / ((1.0 + kc) * (1.0 + kc) * Q)
		b3= (1.0 - (np.tan( wc/2.0) * x0 )) / ( 1.0 + (np.tan(wc/2) * x0 ))
		a[1]=-1.0 * ( 1.0 + b3) * np.cos(wc)
		a[2]=b3
		b[0]=1.0 + (kc * (1.0-b3)/2.0)
		b[1]=-1.0 * (1.0 + b3) * np.cos(wc)
		b[2]=b3 - (kc * (1.0 - b3) /2.0)
		return a,b

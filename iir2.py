

import numpy as np


# calculation 2nd order iir filter process
#
# y[0]= b[0] * x[0]  + b[1] * x[-1] + b[2] * x[-2]
# y[0]= y[0] - a[1] * y[-1] - a[2] * y[-2]
# initial state value is equivalent to zero
# Following code is refer from id:aidiary's Blog page.


class IIR2:
	def __init__(self):
		pass

	def process(self,x,a,b):   
		y = [0.0] * len(x)
		Q = len(a) - 1
		P = len(b) - 1
		for n in range(len(x)):
			for i in range(0, P + 1):
				if n - i >= 0:
					y[n] += b[i] * x[n - i]
			for j in range(1, Q + 1):
				if n - j >= 0:
					y[n] -= a[j] * y[n - j]
		return y

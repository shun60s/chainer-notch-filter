

#
# make sin sweep signal from fl Hz to fh Hz as npoint
# SIN_SWEEP1 is linear frequency sweep
# SIN_SWEEP2 is log frequency sweep

import numpy as np

FS0=48000
FCL=50.0
FCH=1000.0
FC0=200
NPOINT0=10000

class SRC1:
	def __init__(self,fs0=FS0):
		self.fs=fs0


	def SIN_SWEEP1(self,fl=FCL,fh=FCH,npoint=NPOINT0):  # fl,fh must float type 
		fcl=(fl * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		fch=(fh * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		delta0=(fcl - fch) / npoint  # Linear Scale
		ys=np.zeros(npoint)
		fcx=0.0
		for i in range(npoint):
			ys[i]=np.sin(2 * np.pi * fcx)
			fcx += (fcl - delta0  * i)
		return ys

	def SIN_SWEEP2(self,fl=FCL,fh=FCH,npoint=NPOINT0):  # fl,fh must float type 
		fcl=(fl * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		fch=(fh * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		delta1=np.power(fch/fcl, 1.0 / (npoint-1)) # Log Scale
		# print "delta1=", delta1
		ys=np.zeros(npoint)
		fcx=fcl
		fcy=fcl
		for i in range(npoint):
			ys[i]=np.sin(2 * np.pi * fcx)
			fcy = fcy * delta1
			fcx += fcy
		return ys


	def SIN1(self,fc=FC0,npoint=NPOINT0):  # fc must float type 
		fcx=(fc * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		xs=np.linspace(0,npoint-1,npoint)
		ys=np.sin(2 * np.pi * fcx * xs)
		return ys

	def RAND1(self,npoint=NPOINT0):
		# set random seed as fix value, avoid different result every time
		np.random.seed(100)
		ys=np.random.rand(npoint) - 0.5
		return ys




#
# get 2nd order iir filter  frequency response
#
# frequency range is from FCL to FCH
# calcuation points are divided by BAND_NUM as log-scale
# output unit is dB. And normalized frequency by sampling frequency FS0
#
# Soem of following code is refer from id:aidiary's Blog page.

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.9.2 
#  matplotlib 1.5.3


import numpy as np
from matplotlib import pyplot as plt

FS0=48000
FCL=50.0
FCH=1000.0
BAND_NUM=100

class IIR2_FREQ:
	def __init__(self,fcl=FCL,fch=FCH,fs0=FS0,band_num=BAND_NUM,plt_pause0=False):
		self.FCL=fcl
		self.FCH=fch
		self.FS0=fs0
		self.BAND_NUM=band_num
		self.plt_pause=plt_pause0

	def H2(self,f, a, b):
		nume = b[0] + b[1] * np.exp(-2j * np.pi * f) + b[2] * np.exp(-4j * np.pi * f)
		deno = 1 + a[1] * np.exp(-2j * np.pi * f) + a[2] * np.exp(-4j * np.pi * f)
		val = nume / deno
		return np.sqrt(val.real ** 2 + val.imag ** 2)

	def H0(self,a,b):
		amp=[]
		freq=[]
		bands= np.zeros( self.BAND_NUM+1)
		fcl=(self.FCL * 1.0) / self.FS0   #  fc * 1.0 means convert fc to float
		fch=(self.FCH * 1.0) / self.FS0   #  fc * 1.0 means convert fc to float
		delta1=np.power(fch/fcl, 1.0 / (self.BAND_NUM)) # Log Scale
		bands[0]=fcl
		#print "i,band = 0", bands[0] * FS0
		for i in range(1, self.BAND_NUM+1):
			bands[i]= bands[i-1] * delta1
			#print "i,band =", i, bands[i] * self.FS0
		for f in bands:
			amp.append(self.H2(f, a, b))
			#print f
		return np.log10(amp) * 20 ,bands * self.FS0

	def H0disp(self,a,b):
		amp, freq= self.H0(a,b)
		plt.clf()
		plt.plot(freq,amp)
		plt.xlabel('Hz')
		plt.ylabel('dB')
		if self.plt_pause : 
			plt.pause(1.0)
		else:
			plt.show()

	def H0disp2(self,a1,b1,a2, b2):
		amp1, freq1= self.H0(a1,b1)
		amp2, freq2= self.H0(a2,b2)
		plt.clf()
		plt.plot(freq1,amp1, label='initial')
		plt.plot(freq2,amp2, label='target')
		plt.xlabel('Hz')
		plt.ylabel('dB')
		plt.legend()
		if self.plt_pause : 
			plt.pause(5.0)
		else:
			plt.show()

	def H0disp3(self,a0,b0,a1,b1,a2, b2):
		amp0, freq0= self.H0(a0,b0)
		amp1, freq1= self.H0(a1,b1)
		amp2, freq2= self.H0(a2,b2)
		plt.clf()
		plt.plot(freq0,amp0, label='initial')
		plt.plot(freq1,amp1, label='present')
		plt.plot(freq2,amp2, label='target')
		plt.xlabel('Hz')
		plt.ylabel('dB')
		plt.legend()
		plt.title('frequency response')
		if self.plt_pause : 
			plt.pause(5.0)
		else:
			plt.show()

	def disp3(self,a0,b0,a1,b1,a2, b2):
		print "+++ comparison of iir filter coefficient"
		print "   initial  present target"
		print "a[0]", a0[0],a1[0],a2[0]
		print "a[1]", a0[1],a1[1],a2[1]
		print "a[2]", a0[2],a1[2],a2[2]
		print "b[0]", b0[0],b1[0],b2[0]
		print "b[1]", b0[1],b1[1],b2[1]
		print "b[2]", b0[2],b1[2],b2[2]



if __name__ == '__main__':
	
	import iir1
	
	# sample plot pattern 1
	pi0= iir1.IIR1(fs0=48000)
	a1, b1= pi0.CUT(180,-8,8)
	pif0=IIR2_FREQ(fcl=20,fch=20000,fs0=48000,band_num=1024,plt_pause0=True)
	pif0.H0disp(a1,b1)
	
	# sample plot pattern 2
	a1, b1= pi0.CUT(200,-6,8)  # initial
	a2, b2= pi0.CUT(180,-8,8)  # target if known
	pif1=IIR2_FREQ()
	pif1.H0disp2(a1,b1,a2,b2)
	
	
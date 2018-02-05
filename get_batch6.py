


#
# make train data
# frequency range from FCL to FCH is divide by BAND_BUNKATU0 into bandx
# As train data, choice random number as sin frequency per each band
#
#get1:
# Output x is iir filter input data, y is iir filter output data (teacher data)
# The size of [self.band_num, self.seq_num] is batch's kind and length. 
#

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.9.2 
#  matplotlib 1.5.3


import numpy as np
import iir1
import iir2
import src1

FS0=48000
FCL=50.0
FCH=1000.0

SEQUENCE_LEN0=500
BAND_BUNKATU0= 5
NDELAY=0
NPOINT0=10000

FC0=200
GAIN0=-6
Q0=8

FC1=180
GAIN1=-8
Q1=8

class GET_BATCH6:
	def __init__(self, band_num=BAND_BUNKATU0, seq_num=SEQUENCE_LEN0,n_delay=NDELAY, npoint=NPOINT0, fs0=FS0, fc0=FC0, gain0=GAIN0, q0=Q0):
		self.band_num= band_num
		self.seq_num= seq_num
		self.delay= n_delay
		self.fs=fs0
		# design iir filter
		self.pi0= iir1.IIR1(fs0=self.fs)
		self.a1,  self.b1= self.pi0.CUT(fc0,gain0,q0)
		print "...iir filter parameters:"
		print "FC=", fc0
		print "GAIN=", gain0
		print "Q=", q0
		print "a[]=", self.a1
		print "b[]=", self.b1
		# frequency range divide by band_num as log scale
		self.bandx= np.zeros( band_num)
		self.bands= np.zeros( band_num+1)
		fcl=(FCL * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		fch=(FCH * 1.0) / self.fs   #  fc * 1.0 means convert fc to float
		delta1=np.power(fch/fcl, 1.0 / (band_num)) # Log Scale
		self.bands[0]=fcl
		print "...frequency of band for batch"
		print "i,band = 0", self.bands[0] * self.fs
		for i in range(1, band_num+1):
			self.bands[i]= self.bands[i-1] * delta1
			print "i,band =", i, self.bands[i] * self.fs
		# ready for iir filter process
		self.pi2= iir2.IIR2()
		kosuu=self.seq_num + (self.delay + 10)  # 10 is reserve data
		self.xs=np.linspace(0,kosuu -1, kosuu)
		# set random seed as a fixed value
		np.random.seed(100)
		# make sin sweep signal as input and get output to process iir filter
		self.count=0
		self.len=npoint
		self.overflow=0
		pi3=iir2.IIR2()
		if self.len > 0 :
			psrc0=src1.SRC1(fs0=self.fs)
			self.xr= psrc0.SIN_SWEEP2(fl=FCL,fh=FCH,npoint=self.len)
			self.y2b= pi3.process(self.xr, self.a1, self.b1)
			print "making sin sweep input/output was done"

	def get1(self):
		x=np.zeros([self.band_num, self.seq_num])
		y=np.zeros([self.band_num, self.seq_num])
		#choice one point in the band range
		for i in range( self.band_num):
			wx=(self.bands[i+1] - self.bands[i]) * np.random.rand(1) + self.bands[i]
			self.bandx[i]=wx * self.fs
			xsin=np.sin(2 * np.pi * wx * self.xs)
			ys=self.pi2.process(xsin, self.a1, self.b1)
			x[i,:]=xsin[0 : self.seq_num ]
			y[i,:]=ys[ self.delay : self.delay + self.seq_num  ]
		return x,y

	def get_band(self):
		return self.bandx

	def get_a1(self):
		return self.a1

	def get_b1(self):
		return self.b1

	def get_xr(self):
		return self.xr    # return sin sweep input

	def get_y2b(self):
		return self.y2b   # return sin sweep output

	def get2(self):
		x=np.zeros([self.band_num, self.seq_num])
		y=np.zeros([self.band_num, self.seq_num])
		self.overflow=0
		for i in range(0,self.band_num):   
			if (self.count + self.seq_num + self.delay) >= self.len:
				self.count=0
				self.overflow=1
				print "WARNING: reset count as 0 in class GET_BATCH6"
			
			y[i,:]=self.y2b[self.count + self.delay : self.count + self.delay + self.seq_num ]
			x[i,:]=self.xr[self.count: self.count + self.seq_num]
			self.count += self.seq_num
		return x,y,self.overflow
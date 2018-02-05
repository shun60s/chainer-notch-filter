

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.9.2 
#  matplotlib 1.5.3
#  chainer 1.20.0.1

print "CAUTION:  This program is not guaranteed. Please use this at your own risk! "

import numpy as np
from matplotlib import pyplot as plt

import chainer
from chainer import cuda, Function, gradient_check, Variable, FunctionSet, optimizers, serializers, utils

import get_batch6

SEQUENCE_LEN0=  500 
BAND_BUNKATU0= 5
NDELAY=0

# set initial value
FS0=48000
FC0=200
GAIN0=-6
Q0=8
print "...set initial iir filter coefficient "
pg0=get_batch6.GET_BATCH6(band_num=BAND_BUNKATU0, seq_num=SEQUENCE_LEN0,n_delay=NDELAY,npoint=0,fs0=FS0,fc0=FC0, gain0=GAIN0, q0=Q0)
a0=pg0.a1
b0=pg0.b1


# select one of following choice as rnn
import CharRNN1 as rnn   # 3 parameters model. l2_x, l2_h, l3_h,
#import CharRNN2 as rnn   # 4 parameters model. l2_x, l2_h, l3_h, "l5_h"
#import CharRNN3 as rnn   # 5 parameters model.l2_x, l2_h, l3_h, "l5_h ,l4_h"

print "...RNN"
model = rnn.CharRNN()
#optimizer = optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)  # default of optimzaier Admam
optimizer = optimizers.Adam(alpha=0.0001)  # Change: optimzaier Admam alpha value from default
optimizer.setup(model)
state = rnn.make_initial_state( batchsize=BAND_BUNKATU0 )

rnn.set_para(model, pg0.a1, pg0.b1)


#---
print "...set target iir filter coefficient"
FC1=180
GAIN1=-8
Q1=8

NPOINT0=5000  # sin sweep signal length

pg0=get_batch6.GET_BATCH6(band_num=BAND_BUNKATU0, seq_num=SEQUENCE_LEN0,n_delay=NDELAY,npoint=NPOINT0,fs0=FS0,fc0=FC1, gain0=GAIN1, q0=Q1)
a1=pg0.a1
b1=pg0.b1

loss   = Variable(np.zeros((), dtype=np.float32))
losses =[]

NUMBER_ITERATION=501

for i in range(NUMBER_ITERATION):
	
	x,y = pg0.get1()  # get train data
	loss, state =  rnn.compute_loss(model, x, y, state)  # do one sequence while batch bands
	model.cleargrads()
	loss.backward()
	optimizer.update()
	
	losses.append(loss.data /(SEQUENCE_LEN0 * 1.0))  # total loss  while one BAND_BUNKATU0
	
	state = rnn.make_initial_state( batchsize=BAND_BUNKATU0 )  	# clear for next batch-sequence-input
	
	if i%20==0:
		plt.plot(losses,"b")
		plt.yscale('log')
		plt.title('loss')
		plt.pause(1.0)
		print "loss.data (%06d)="%i, loss.data / (SEQUENCE_LEN0 * 1.0)
	##if i%100==0:  # save model parameter in the directory model20  every 100 
	##	serializers.save_npz('model20/%06d_my.model.npz'%i, model)


# load model parameter
##serializers.load_npz('model20/000500_my.model.npz',model)

# set True if plot is pause for some seconds
PLOT_PAUSE=True

# Get last parameter and performance check
print "...last parameters"
rnn.show_para(model)


import iir2_freq

FCL=50.0
FCH=1000.0
BAND_NUM=100

pif0=iir2_freq.IIR2_FREQ(fcl=FCL,fch=FCH,fs0=FS0,band_num=BAND_NUM,plt_pause0=PLOT_PAUSE)

a2,b2=rnn.set_coefficient(model)
pif0.H0disp3(a0,b0,a2,b2,a1,b1)
pif0.disp3(a0,b0,a2,b2,a1,b1)

x,y = pg0.get1()
model.train=False
loss, data0 =  rnn.compute_loss(model, x, y, state)  # do one sequence while batch bands
y10=np.zeros( y.size )
y11=np.zeros( y.size )
rows, cols = y.shape   # BAND_BUNKATU0,SEQUENCE_LEN0
for i in range( rows ):
	y10[ cols * i : cols * (i+1)] = y[i,:]
	y11[ cols * i : cols * (i+1)] = data0[i,:]

plt.clf()
plt.plot(y10,'b',label='iir filter output')
plt.plot(y11,'r',label='RNN output')
plt.legend()
plt.title('comparison per each bands')
if PLOT_PAUSE :
	plt.pause(5.0)
else:
	plt.show()


print "...checking by sin sweep input/output signal"
y2=np.zeros(NPOINT0)

model.train = False
state = rnn.make_initial_state( batchsize=1)  # one by one call, not batch type

for i in range(NPOINT0):
	x1 = chainer.Variable(np.asarray( [ pg0.get_xr()[i] ] , dtype=np.float32)[:, np.newaxis] )
	dummy=chainer.Variable(np.asarray( [0] , dtype=np.float32) [:, np.newaxis] )
	state, yout= model.forward_one_step(x1, dummy, state)
	y2[i]= yout.data

plt.clf()
plt.plot(pg0.get_y2b(),'b',label='iir filter output')
plt.plot(y2,'r',label='RNN output')
plt.legend()
plt.title('sin sweep output')
if PLOT_PAUSE :
	plt.pause(5.0)
else:
	plt.show()

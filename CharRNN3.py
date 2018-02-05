

# Check version
#  Python 2.7.12 on win32 (Windows version)
#  numpy 1.9.2 
#  chainer 1.20.0.1

import numpy as np

import chainer
from chainer import cuda, Function, gradient_check, Variable, FunctionSet, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L



SEQUENCE_LEN0= 500 #  data quantity per one optimizer.update() call
BAND_BUNKATU0= 5   #  bands quantity (= n-batch), select train random data per each bands 

IN_SIZE0=1  # input dimension: 1
HIDDEN_UNITS0=1  #  parallel dimension: 1
OUT_SIZE0=1   # output dimension: 1



#   5 parameters. l2_x, l2_h, l3_h, "l5_h ,l4_h" are used for iir filter coefficient learning.
#   1 parameter l6 is amplitude factor.


class CharRNN(FunctionSet):
	def __init__(self, in_size=IN_SIZE0, out_size=OUT_SIZE0, hn_units=HIDDEN_UNITS0, train=True):
		super(CharRNN, self).__init__(
			# 1st coefficient fixed to 1
			l2_x = L.Linear(hn_units, hn_units, nobias=True),   
			l2_h = L.Linear(hn_units, hn_units, nobias=True),
			l3_h = L.Linear(hn_units, hn_units, nobias=True),
			#l3_x = L.Linear(hn_units, hn_units, nobias=True),
			l4_h = L.Linear(hn_units, hn_units, nobias=True),
			l5_h = L.Linear(hn_units, hn_units, nobias=True),
			l6=L.Linear(hn_units, out_size, nobias=True),      
		)
		# set random seed as fix value, avoid different result every time
		np.random.seed(100)
		for param in self.parameters:
			param[:] = np.random.uniform(-0.5, 0.5, param.shape)
		self.train=train
		self.hn_units=hn_units
	#def __call__(self, x, t):

	def forward_one_step(self, x, y, state, train=True, dropout_ratio=0.0):
		h1 = x   # h1 = F.liner(x, Unit1)  No need in Super declare section
		c2   = self.l2_x(F.dropout(h1, ratio=dropout_ratio, train=train)) + self.l2_h(state['h2']) + self.l3_h(state['h3'])
		h3=state['h2']
		h2=h1
		c3   = c2 + self.l4_h(state['h4']) + self.l5_h(state['h5'])
		#c3= F.tanh(c3)
		h5=state['h4']
		h4=c3 
		t = self.l6(c3)
		
		state   = { 'h2': h2 , 'h3': h3, 'h4': h4 , 'h5': h5}
		
		self.loss = F.mean_squared_error(y, t)   # heikin 2 zyou gosa [ siguma (xi-yi)^2 ] / kossu
		self.prediction=t
		
		if self.train:
			return state, self.loss
		else:
			return state, self.prediction


def compute_loss(model, x, y, state):
	state0=state
	loss = 0
	rows, cols = x.shape   # BAND_BUNKATU0,SEQUENCE_LEN0
	batch_size= rows
	length_of_sequence = cols
	if not model.train :
		datax0= np.zeros([rows,cols])
	for i in range(cols - 1):
		x1 = chainer.Variable(np.asarray([x[j, i ] for j in range(rows)], dtype=np.float32)[:, np.newaxis])
		t1 = chainer.Variable(np.asarray([y[j, i ] for j in range(rows)], dtype=np.float32)[:, np.newaxis])
		if model.train:
			state0, loss0 =model.forward_one_step(x1, t1, state0, dropout_ratio=0.0)
			loss += loss0
		else:  # model.train is False
			state0, data0 =model.forward_one_step(x1, t1, state0, dropout_ratio=0.0)
			datax0[:,i]=data0.data.reshape(rows)
	if model.train:
		return loss, state0
	else:
		return loss, datax0


def make_initial_state(hn_units=HIDDEN_UNITS0, batchsize=BAND_BUNKATU0, train=True):
    return {name: Variable(np.zeros((batchsize, hn_units), dtype=np.float32),volatile=not train)
            for name in (  'h2',  'h3' , 'h4', 'h5')}

def scalar(v):
	#return scalar of valiable
	return v.data.ravel()[0]

def scalar2(v):
	#return scalar of valiable
	return v.data.ravel()[:]

def set_para(model, a1, b1):
	# initialize  model as same as IIR filter coefficient of a1, b1
	
	#model.l1.W=chainer.Variable(np.asarray( [1] , dtype=np.float32) [:, np.newaxis] )
	#print scalar2( model.l1.W)
	model.l6.W=chainer.Variable(np.asarray( [1] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l6.W)
	
	print ""
	
	model.l2_x.W=chainer.Variable(np.asarray( [ b1[0]] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l2_x.W)
	model.l2_h.W=chainer.Variable(np.asarray( [ b1[1]] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l2_h.W)
	model.l3_h.W=chainer.Variable(np.asarray( [ b1[2]] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l3_h.W)
	
	#model.l3_x.W=chainer.Variable(np.asarray( [ a1[0]] , dtype=np.float32) [:, np.newaxis] )
	#print scalar2( model.l3_x.W)
	model.l4_h.W=chainer.Variable(np.asarray( [ a1[1] * -1.0] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l4_h.W)
	model.l5_h.W=chainer.Variable(np.asarray( [ a1[2] * -1.0] , dtype=np.float32) [:, np.newaxis] )
	print scalar2( model.l5_h.W)

def show_para(model):
	print "...show_parametrs:"
	#print scalar2( model.l1.W)
	print "l1.W = 1"
	print scalar2( model.l6.W)
	
	print ""
	
	print scalar2( model.l2_x.W)
	print scalar2( model.l2_h.W)
	print scalar2( model.l3_h.W)
	
	#print scalar2( model.l3_x.W)
	print "l3_x.W = 1 fixed"
	print scalar2( model.l4_h.W)
	#print "l4_h.W = - l2_h.W"
	print scalar2( model.l5_h.W)
	#print "l5_h.W = -l2_x.W - l3_h.W + 1 "

def set_coefficient(model):
	a2 = [0.0] * 3
	b2 = [0.0] * 3
	b2[0]=scalar( model.l2_x.W)
	b2[1]=scalar( model.l2_h.W)
	b2[2]=scalar( model.l3_h.W)
	a2[0]=1.0
	a2[1]= -1.0 * scalar( model.l4_h.W)
	a2[2]= -1.0 * scalar( model.l5_h.W)
	return a2,b2
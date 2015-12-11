import numpy as np
import sys
import csv
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

def flip_bits(x, percent):
	#print x
	plot(x, "pre-flip")
	n = len(x)

	K = int(percent*n)

	for k in range(0,K):
		r = random.random()
		rand_index = int(math.floor(r*n))
		
		if x[rand_index] == -1:
			x[rand_index] = 1

		elif x[rand_index] == 1:
			x[rand_index] = -1

	#print x
	plot(x,"post-flip")
	return x

def plot(x, info):
	n = (len(x)**.5)
	pic = np.reshape(x, (n,n))
	plt.suptitle(info, fontsize=18)
	plt.imshow(pic, cmap = "binary")
	plt.show()

# Creates a set of weights between nodes in a Hopfield network whose size is based on the length of the rows in the input data X.
# X is a numpy.array of shape (R, C). Values in X are drawn from {+1,-1}. Each row is a single training example. 
# X(3,:) would be the 3rd training example.
# W is a C by C numpy array of weights, where W(a,b) is the connection weight between nodes a and b in the Hopfield net after training.
# Here, C = number of nodes in the net = number of columns in X

def train_hopfield(X):
	print "train_hopfield"

	#make weighted connection matrix
	n = len(X[0])
	C = len(X[:,0])

	W = np.zeros((n,n))
	
	for i in range(0,n):
		for j in range(0,n):
			if i==j:
				continue

			sum = 0
			for c in range(0,C):
				sum += X[c][i]*X[c][j]

			W[i][j] = sum
	#plt.imshow(W)
	plt.show()
	print "W= " 
	print 	str(W)

	return W

# Takes a Hopfield net W and an input vector x and runs the Hopfield network until it converges.
# x, the input vector, is a numpy vector of length C (where C is the number of nodes in the net). 
# This is a set of activation values drawn from the set {+1, -1}
# W is a C by C numpy array of weights, where W(a,b) is the connection weight between nodes a and b in the Hopfield net.
# Here, C = number of nodes in the net. s is a numpy vector of length C (number of nodes in the net) containing the final activation of the net. 
# It is the result of giving x to the net as input and then running until convergence.
def use_hopfield(W,x):
	print "use_hopfield"
	#print x
	plot(x, "x to be matched via hopfield network")

	s = x[:]

	for iteration in range(0,10):
		s_old = s[:]
		for i in range(0,len(x)):

			col_W = W[:,i]
			
			val = sum(s * col_W)
			update = -1

			if val>= 0:
				update = 1

			s[i] = update
			print "iteration " + str(iteration) + ": s["+str(i)+"]= " + str(x[i]) + "; val =" + str(val) +"; new s[i] = " + str(update)
	
		plot(s, "s - after iteration "+str(iteration))
		if (s == s_old):
			print "s-convergence: " + str(s)
			break
		else:
			print "s-current: " + str(s)

	return s

def convert_zeros_to_negOnes(x):

	mod_x = []

	for i in range(0,len(x)):
		if x[i]==0:
			mod_x.append(-1)
		else:
			mod_x.append(1)

	return mod_x

def main():
	# read in the csv file
	rfile = sys.argv[1]

	#csvfile = open(rfile, 'rb')
	#dat = csv.reader(csvfile, delimiter=' ')
	# X = []
	# Y = []

	X = np.loadtxt(rfile, delimiter = ' ')
	#print X
	#print X
	
	#C = 1

	Classes = []

	for i in range(0,3):
		one_image = X[i]
		one_image = one_image[0:256]
		one_image = convert_zeros_to_negOnes(one_image)

		#plot(one_image)

		if i!=2:
			Classes.append(one_image)
		
		if i==2:
			x = one_image[:]

	X = np.array(Classes)
	#flip_bits(x,.2)

	W = train_hopfield(X)

	s = use_hopfield(W,x)

if __name__ == "__main__":
	main()
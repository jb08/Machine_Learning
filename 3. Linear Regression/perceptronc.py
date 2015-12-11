import sys
import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt

#helper function used as sign function in Perceptron Algorithm
def classify(h_k):
	if h_k>=0:
		return 1
	else:
		return -1

#Output graphs added to help user understand data better
def perceptronc(w_init, X, Y):
	#figure out (w, k) and return them here. w is the vector of weights, k is how many iterations it took to converge.

	for i in range(0,X.shape[0]):
		X[i] = abs(X[i]-1.5)

	#Used to build starting graph
	X_pos = []
	X_neg = []

	for i in range(0,Y.shape[0]):
		if Y[i]>0:
		 	X_pos.append(X[i])
		else:
		 	X_neg.append(X[i])

	zeros_X_pos = np.zeros(len(X_pos))
	zeros_X_neg = np.zeros(len(X_neg))

	plt.title("Transformed Data to be Classified: |x-1.5|")
	plt.xlabel("X-Value")
	plt.ylabel("Univariate attribute")
	plt.plot(X_pos,zeros_X_pos, "rs")
	plt.plot(X_neg,zeros_X_neg, "bs")
	plt.show()
	x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)

	#setup while loop
	b = np.zeros((X.shape[0],2))

	for i in range(0,X.shape[0]):
		b[i,0] = 1
		b[i,1] = X[i]
	w = np.zeros((1,2))
	X = b
	k = 0
	epoch_count = 0
	learning_rate = .2
	no_updates_made_since_last_epoch = True
	incorrect_classificiations_exist = True

	#iterate until successful classification or k > 2000 (Failure)
	while(incorrect_classificiations_exist):
		h_k = round(X[k,0]*w[0,0] + X[k,1]*w[0,1],3)
		sgn = classify(h_k)

		#if data misclassified, modify w vector
		if sgn != Y[k]:
			w[0,0] = w[0,0] + Y[k]*learning_rate*1
			w[0,1] = w[0,1] + Y[k]*learning_rate*X[k,1]
			no_updates_made_since_last_epoch = False
			print "misclassified-> new w : " + str(w)

		#iterate counter
		k = (k+1)

		#check status of loop - add to epoch_count, check for success
		if k == X.shape[0]:
			k = 0
			epoch_count = epoch_count + 1

			if no_updates_made_since_last_epoch:
				print "Algorithm aborted: (w,k) = (" + str(w) + " , " + str(epoch_count) + ")"

				line = []
				t = np.arange(np.amin(X)-2, np.amax(X)+2, .5)
				for x in t:
					line.append(w[0,0] + w[0,1]*x)

				plt.title("Perceptron Algorithm Worked: W= "+ str(w))
				plt.xlabel("X-Value")
				plt.ylabel("Univariate attribute")
				plt.plot(t,line, "g--")
				plt.plot(X_pos,zeros_X_pos, "rs")
				plt.plot(X_neg,zeros_X_neg, "bs")
				plt.show()

				return (w,epoch_count)
			else:
				no_updates_made_since_last_epoch = True
				if epoch_count > 2000:
					print "Algorithm aborted: k exceeded 2000 (FAILURE)"
					return (w, epoch_count)

def main():
	rfile = sys.argv[1]
	
	#read in csv file into np.arrays X1, X2, Y1, Y2
	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X1 = []
	Y1 = []
	X2 = []
	Y2 = []
	for i, row in enumerate(dat):
		if i > 0:
			X1.append(float(row[0]))
			X2.append(float(row[1]))
			Y1.append(float(row[2]))
			Y2.append(float(row[3]))
	X1 = np.array(X1)
	X2 = np.array(X2)
	Y1 = np.array(Y1)
	Y2 = np.array(Y2)
	w_init = 0 # INTIALIZE W_INIT
	#perceptronc(w_init, X1, Y1)
	perceptronc(w_init, X2, Y2)

if __name__ == "__main__":
	main()
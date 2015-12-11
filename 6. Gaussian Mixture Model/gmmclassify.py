import numpy as np
import sys
import csv
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats

def probability_x_given_c(x, mu, sigmasq, wt):

	K = len(mu)

	prob = 0

	for k in range(0,K):
			norm = stats.norm(mu[k],math.sqrt(sigmasq[k]))
			p = wt[k] * norm.pdf(x)
			#print p
			prob += p

	return prob

def gmmclassify(X, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1):
	
	n = len(X)
	Y_hat = []
	class_1s = []
	class_2s = []

	for i in range(0,n):
		prob_1 = probability_x_given_c(X[i],mu1, sigmasq1, wt1)
		prob_2 = probability_x_given_c(X[i],mu2, sigmasq2, wt2)

		weighted_prob_1 = p1*prob_1
		weighted_prob_2 = (1-p1)*prob_2

		if (weighted_prob_1 > weighted_prob_2):
		#if (prob_1 > prob_2):
			Y_hat.append(1)
			class_1s.append(X[i])
		else:
			Y_hat.append(2)
			class_2s.append(X[i])

	return Y_hat, class_1s, class_2s

def compare_results(Y_hat, Y):
	
	n = len(Y_hat)
	accuracy_count = 0.0

	for i in range(0,n):
		predicted_class = Y_hat[i]
		actual_class = Y[i]

		if predicted_class==actual_class:
			accuracy_count +=1
	
	accuracy_rate = accuracy_count / n
	return accuracy_rate

def plot(X_test, Y_test, class_1s, class_2s, mu1, sigmasq1, wt1, mu2, sigmasq2, wt2):

	class_1 = X_test[np.nonzero(Y_test ==1)[0]]
	class_2 = X_test[np.nonzero(Y_test ==2)[0]]

	bins = 50 # the number 50 is just an example.
	# plt.subplot(2,1,1)
	
	plt.scatter(class_1s, [0]*len(class_1s), c='r', s=80)
	plt.scatter(class_2s, [0]*len(class_2s), c='y', s=80)

	count, bins_1, ignored = plt.hist(class_1, bins, normed = True)
	count, bins_2, ignored = plt.hist(class_2, bins, normed = True)

	color = ['r', 'y']
	plt.axis([-40, 80, -.02, .15])
	#plt.axis('auto')

	K = len(mu1)
	
	for k in range(0,K):

		mu_k = mu1[k]
		sigma = sigmasq1[k]**.5

		plt.plot(bins_1, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins_1 - mu_k)**2 / (2 * sigma**2) ), 
			linewidth=2, color=color[0])

	K = len(mu2)
	
	for k in range(0,K):

		mu_k = mu2[k]
		sigma = sigmasq2[k]**.5
		
		plt.plot(bins_2, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins_2 - mu_k)**2 / (2 * sigma**2) ), 
			linewidth=2, color=color[1])

	plt.title("Class 1 (green) and Class 2 (blue) data")
	plt.xlabel("Data Values")
	plt.ylabel("Percentage of Values")

	plt.show()

def main():
	# read in the csv file
	
	rfile = sys.argv[1]

	csvfile = open(rfile, 'rb')
	dat = csv.reader(csvfile, delimiter=',')
	X = []
	Y = []

	# put the x coordinates in the list X, the y coordinates in the list Y
	for i, row in enumerate(dat):

		if i > 0:
			X.append(float(row[0]))
			Y.append(float(row[1]))
	X = np.array(X)
	Y = np.array(Y)

	#class1 = X[np.nonzero(Y ==1)[0]]
	#class2 = X[np.nonzero(Y ==2)[0]]

	#data from gmmest.py
	mu1 = [9.77,29.58]
	sigmasq1 = [21.92,9.78]
	wt1 = [0.60,0.40]
	mu2 = [-24.82, -5.06, 49.62]
	sigmasq2 = [7.95, 23.32, 100.02]
	wt2 = [0.2,0.5,0.3]
	p1 = .33
	
	Y_hat, class_1s, class_2s = gmmclassify(X,mu1, sigmasq1, wt1, mu2, sigmasq2, wt2, p1)
	#accuracy_rate = compare_results(Y_hat, Y)

	plot(X,Y, class_1s, class_2s, mu1,sigmasq1,wt1, mu2, sigmasq2, wt2)

if __name__ == "__main__":
	main()
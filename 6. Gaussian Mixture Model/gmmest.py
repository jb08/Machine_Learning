import numpy as np
import sys
import csv
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats


#maximization step
def maximization_step(X,responsibilities_j,mu):

	gamma = sum(responsibilities_j)

	#updated weight
	n = len(X)
	new_wt = gamma/n

	#updated mean
	new_mu_numerator = 0

	for i in range(0,n):
		resp_ji = responsibilities_j[i]
		val = X[i]
		#print str(resp_i) + ", " + str(val)
		new_mu_numerator += (resp_ji*val)

	new_mu = new_mu_numerator/gamma

	new_sigmasq_numerator = 0

	#updated variance
	for i in range(0,n):
		resp_ji = responsibilities_j[i]
		val = X[i]

		prod = resp_ji*(val-mu)**2
		new_sigmasq_numerator +=prod

		#print str(resp_i) + ", " + str(val) + ", " + str(prod)

	new_sigmasq = new_sigmasq_numerator/gamma

	return new_wt, new_mu, new_sigmasq

def expectation_step(X, mu, sigmasq, wt):

	n = len(X)
	K = len(mu)

	responsibilities_numerators = []

	for k in range(0,K):
		norm = stats.norm(mu[k],math.sqrt(sigmasq[k]))
		resp_k_numerators_only = []

		for i in range(0,n):

			x = X[i]
			p = norm.pdf(x)

			resp_k_numerators_only.append(wt[k] * p)
			
		#print resp_k_numerators_only
		responsibilities_numerators.append(resp_k_numerators_only)

	responsibilities_denominators = []

	for i in range(0,n):

		denom_i = 0

		for k in range(0,K):
			denom_i += responsibilities_numerators[k][i]

		responsibilities_denominators.append(denom_i)

	responsibilities = []

	for k in range(0,K):

		responsibilities_k = []

		for i in range(0,n):
			numerator = responsibilities_numerators[k][i]
			denominator = responsibilities_denominators[i]
			responsibilities_k.append(numerator/denominator)

		responsibilities.append(responsibilities_k)

	return responsibilities

# % Input
	# % - X : N 1-dimensional data points (a 1-by-N vector)
	# % - mu_init : initial means of K Gaussian components (a 1-by-K vector)
	# % - sigmasq_init: initial variances of K Gaussian components (a 1-by-K vector)
	# % - wt_init : initial weights of k Gaussian components (a 1-by-K vector that sums to 1)
	# % - its : number of iterations for the EM algorithm
def gmmest(X,mu_init,sigmasq_init,wt_init,its):

	verbose = True
	plot_on = False
	stop_upon_convergence = False

	n = len(X)
	K = len(mu_init)
	mu = mu_init
	sigmasq = sigmasq_init
	wt = wt_init
	old_L = float("inf")
	L = 0
	convergence = 1
	log_Lik_by_it = []

	for it in range(0,its):

		if stop_upon_convergence and abs(L-old_L) < convergence:
			break

		if verbose:
			print "iteration " + str(it) + ": "

		#expectation step
		responsibilities = expectation_step(X, mu, sigmasq, wt)

		#maximization step
		for k in range(0,K):
			new_wt, new_mu, new_sigmasq = maximization_step(X,responsibilities[k],mu[k])
			
		
			if verbose:
				#print '  k:{} old_wt:{} old_mu:{} old_sigmasq:{} new_wt:{} new_mu:{} new_sigmasq:{}'\
				#.format(k, round(wt[k],2), round(mu[k],2), round(sigmasq[k],2), round(new_wt,2), round(new_mu,2), round(new_sigmasq,2))

				print '  k:{} new_wt:{} new_mu:{} new_sigmasq:{}'\
				.format(k, round(new_wt,2), round(new_mu,2), round(new_sigmasq,2))

			wt[k] = new_wt
			mu[k] = new_mu
			sigmasq[k] = new_sigmasq

		#calculate log likelihood

		old_L = L
		L = 0

		for i in range(0,n):

			sum_of_probabilities = 0

			for k in range (0,K):
				norm = stats.norm(mu[k],math.sqrt(sigmasq[k]))
				probability = norm.pdf(X[i])
				weighted_probability = wt[k] * probability

				sum_of_probabilities += weighted_probability

			log_sum = math.log(sum_of_probabilities)

			L += log_sum

		log_Lik_by_it.append(L)

		if verbose:
			print "  log likelihood: " + str(round(L,2))

	if plot_on:

		plt.title("Log likelihood during EM Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Log likelihood")
		plt.plot(log_Lik_by_it, "b")
		plt.show()

	#print log_Lik_by_it

	# % Output
		# % - mu : means of Gaussian components (a 1-by-K vector)
		# % - sigmasq : variances of Gaussian components (a 1-by-K vector)
		# % - wt : weights of Gaussian components (a 1-by-K vector, sums to 1)
		# % - L : log likelihood

	return mu, sigmasq, wt, L, log_Lik_by_it

def plot(X_test, Y_test, mu_1, sigmasq_1, mu_2, sigmasq_2):

	class1 = X_test[np.nonzero(Y_test ==1)[0]]
	class2 = X_test[np.nonzero(Y_test ==2)[0]]
	bins = 50 # the number 50 is just an example.
	# plt.subplot(2,1,1)
	count, bins, ignored = plt.hist(class1, bins, normed = True)

	K = len(mu_1)
	color = ['r', 'g']

	for k in range(0,K):

		mu_k = mu_1[k]
		sigma = sigmasq_1[k]**.5

		plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins - mu_k)**2 / (2 * sigma**2) ), 
			linewidth=2, color=color[k])

	# plt.subplot(2,1,2)
	# count, bins, ignored = plt.hist(class2, bins, normed = True)

	# K = len(mu_2)

	# for k in range(0,K):

	# 	mu_k = mu_2[k]
	# 	sigma = sigmasq_2[k]**.5

	# 	plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
	# 		np.exp( - (bins - mu_k)**2 / (2 * sigma**2)), 
	# 		linewidth=2, color=color[k])

	plt.show()


def get_variance(class_data):

	sample_mu = sum(class_data)/len(class_data)
	variance = 0

	for i in range(0, len(class_data)):
		variance += (class_data[i]-sample_mu)**2

	return variance

def get_init_mu(class_data, K):

	n = len(class_data)
	mu_init = []

	for k in range(0,K):
		x = random.random()
		rand_index = math.floor(x*n)
		mu_init.append(class_data[rand_index])

	return mu_init

def plot_one_class(X_test, Y_test, mu, sigmasq, class_val):

	class_1or2 = X_test[np.nonzero(Y_test ==class_val)[0]]
	bins = 50 # the number 50 is just an example.
	# plt.subplot(2,1,1)
	count, bins, ignored = plt.hist(class_1or2, bins, normed = True)

	K = len(mu)
	color = ['r', 'g', 'y']

	for k in range(0,K):

		mu_k = mu[k]
		sigma = sigmasq[k]**.5

		plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
			np.exp( - (bins - mu_k)**2 / (2 * sigma**2) ), 
			linewidth=2, color=color[k%len(color)])

	plt.title("Histogram of class and GMM normal distributions")
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

	class1 = X[np.nonzero(Y ==1)[0]]
	class2 = X[np.nonzero(Y ==2)[0]]

	run_class_1 = False

	if run_class_1:

		#if class 1
		K = 2
		variance = get_variance(class1)

		sigmasq_init = [variance] * K
		mu_init = get_init_mu(class1,K)

		#sigmasq_init = [1,1]
		#mu_init = [30,10]

		# print mu_init

		#mu_init = [10-5,30-5]

		wt_init = [1.0/K] * K
		its = 20
		class_val = 1

		runs = 1

		mu = 0
		sigmasq = 0
		wt = 0
		L = float("-inf")
		L_list = []
		mu_init_save = 0

		for x in range(0,runs):
			mu_init_maybe_save = mu_init[:]
			mu_run, sigmasq_run, wt_run, L_run, L_list_run = gmmest(class1, mu_init, sigmasq_init,wt_init,its)

			if (L_run> L):
				mu = mu_run
				sigmasq = sigmasq_run
				wt = wt_run
				L = L_run
				L_list = L_list_run
				mu_init_save = mu_init_maybe_save

		# print mu
		# print sigmasq
		# print wt
		# print L
		# print L_list
		# print mu_init_save

		plt.title("Log likelihood during EM Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Log likelihood")
		plt.plot(L_list, "b")
		plt.show()


	else:
		#if class 2
		K = 3
		variance = get_variance(class2)
		sigmasq_init = [variance] * K
		mu_init = get_init_mu(class2,K)
		wt_init = [1.0/K] * K
		its = 20
		class_val = 2

		#mu_init = [-30,-10,50]
		#sigmasq_init = [100] * K

		runs = 1

		mu = 0
		sigmasq = 0
		wt = 0
		L = float("-inf")
		L_list = []
		mu_init_save = 0

		for x in range(0,runs):
			mu_init_maybe_save = mu_init[:]
			mu_run, sigmasq_run, wt_run, L_run, L_list_run = gmmest(class2, mu_init, sigmasq_init,wt_init,its)

			if (L_run> L):
				mu = mu_run
				sigmasq = sigmasq_run
				wt = wt_run
				L = L_run
				L_list = L_list_run
				mu_init_save = mu_init_maybe_save


		plt.title("Log likelihood during EM Iterations")
		plt.xlabel("Iterations")
		plt.ylabel("Log likelihood")
		plt.plot(L_list, "b")
		plt.show()

		# print mu
		# print sigmasq
		# print wt
		# print L
		# print L_list
		# print mu_init_save

	plot_one_class(X,Y,mu, sigmasq,class_val)

if __name__ == "__main__":
	main()
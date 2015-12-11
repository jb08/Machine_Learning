#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Jason Brown
from fractions import Fraction
from operator import mul
import scipy.stats as stats
import sys
import numpy as np
import os
import shutil

def sanitize_word(word):

	word = word.lower()
	num = ord(word[-1])
	ends_in_symbol = (num < 48 and num > 31) or (num < 65 and num > 57)

	if(ends_in_symbol):
		word = word[0:-1]

	return word

def is_not_real_word(word):
	
	max_symbols_allowed = 8

	symbol_count = 0

	for char in word:
		num = ord(char)
	
		#@ and . are not considered symbols here
		is_symbol = (num < 46 and num > 31) or (num < 64 and num > 57)
		if is_symbol:
			symbol_count +=1

	return symbol_count >= max_symbols_allowed

def incr_num_and_denom(numerator, denominator):
	
	return Fraction(numerator + 1, denominator + 1).limit_denominator(max_denominator=denominator+1)

def incr_all_fracs(words, sizeofspam, sizeofham):

	for k in words:
		s_numerator = Fraction(words[k]['spam']).numerator
		h_numerator = Fraction(words[k]['ham']).numerator

		words[k]['spam'] = incr_num_and_denom(s_numerator, sizeofspam)
		words[k]['ham'] = incr_num_and_denom(h_numerator, sizeofham)
	

def parse(text_file):
	#This function parses the text_file passed into it into a set of words. 
	#Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)

def makedictionary(spam_directory, ham_directory, dictionary_filename):
	#Making the dictionary. 
	
	spam = [f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f)) and f!= ".DS_Store"]
	ham = [f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f)) and f!= ".DS_Store"]

	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))

	words = {}

	#These for loops walk through the files and construct the dictionary. 
	#The dictionary, words, is constructed so that words[word]['spam'] 
	#gives the probability of observing that word, given we have a spam 
	#document P(word|spam), and words[word]['ham'] gives the probability of 
	#observing that word, given a ham document P(word|ham). Right now, all 
	#it does is initialize both probabilities to 0. 
	#TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).

	for s in spam:
		for word in parse(open(spam_directory + s)):

			if is_not_real_word(word):
				continue

			word = sanitize_word(word)

			if word not in words:
				words[word] = {'spam': Fraction(1,len(spam)), 'ham': Fraction(0)}
			else:
				prob = Fraction(words[word]['spam']).limit_denominator(max_denominator=len(spam))

				words[word]['spam'] = Fraction(prob.numerator + 1,len(spam))

	for h in ham:
		for word in parse(open(ham_directory + h)):

			if is_not_real_word(word):
				continue
			
			word = sanitize_word(word)

			if word not in words:
				words[word] = {'spam': Fraction(0), 'ham': Fraction(1,len(ham))}
			else:
				prob = Fraction(words[word]['ham']).limit_denominator(max_denominator=len(ham))

				words[word]['ham'] = Fraction(prob.numerator + 1,len(ham))
	
	#Write it to a dictionary output file.

	incr_all_fracs(words, len(spam), len(ham))

	writedictionary(words, dictionary_filename)

	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability
	#is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. 
	#You need to update it to make it use the dictionary and the content of the mail. 
	#Here is where your naive Bayes classifier goes.

	logs_spam = []
	logs_spam.append(np.log2(float(spam_prior_probability)))
	
	logs_ham = []
	logs_ham.append(np.log2(float(1-spam_prior_probability)))
	
	for word in content:

		word = word.lower()
		
		if word not in dictionary:
			continue

		#add spam probabilities
		prob_spam = dictionary[word]['spam']
		logs_spam.append(np.log2(float(prob_spam)))
		
		#add ham probabilities
		prob_ham = dictionary[word]['ham']
		logs_ham.append(np.log2(float(prob_ham)))

	likelihood_spam_L = sum(logs_spam)
	
	likelihood_ham_L = sum(logs_ham)
	
	if likelihood_spam_L >= likelihood_ham_L:
		
		#print "sum log ham: " + str(round(likelihood_ham_L)) + "; sum log spam: " + str(round(likelihood_spam_L)) + " -> SPAM"

		return True
	else:
 
		print "sum log ham: " + str(round(likelihood_ham_L)) + "; sum log spam: " + str(round(likelihood_spam_L)) + " -> HAM"

		return False

def spamsort(mail_directory, spam_directory, ham_directory, dictionary, spam_prior_probability):
	
	mail = [f for f in os.listdir(mail_directory) if os.path.isfile(os.path.join(mail_directory, f)) and f!= ".DS_Store"]
	for m in mail:
		content = parse(open(mail_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			shutil.copy(mail_directory + m, spam_directory)
		else:
			shutil.copy(mail_directory + m, ham_directory)

if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, 
	#and a mail_directory that is filled with unsorted mail on the command line. It will create two 
	#directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will 
	#show up in this directories according to the algorithm you developed.

	post_experiment = True

	if post_experiment:
			#30 IIDs in each test, numbers represent # of correct picks
			method_prior_probability = [21] * 14
			method_bayes = [29, 26, 18, 29, 27, 26, 30, 28, 29, 27, 30, 28, 28, 28]

			print "Experiment Results:"
			print "Correct classifications out of test set size of 30:"
			print "method_bayes: " + str(method_bayes)
			print "method_prior_probability: " + str(method_prior_probability)

			paired_sample = stats.ttest_rel(method_bayes, method_prior_probability)

			print "The p-value is %.3f. This rejects the null hypothesis (described further in write-up)" % paired_sample[1]

	training_spam_directory = sys.argv[1]
	training_ham_directory = sys.argv[2]
	
	test_mail_directory = sys.argv[3]
	test_spam_directory = 'sorted_spam'
	test_ham_directory = 'sorted_ham'
	
	if not os.path.exists(test_spam_directory):
		os.mkdir(test_spam_directory)
	if not os.path.exists(test_ham_directory):
		os.mkdir(test_ham_directory)
	
	dictionary_filename = "dictionary.dict"
	
	#create the dictionary to be used
	dictionary, spam_prior_probability = makedictionary(training_spam_directory, training_ham_directory, dictionary_filename)

	print 
	print "Current training set Spam prior probability: " + str(spam_prior_probability)
	#sort the mail
	spamsort(test_mail_directory, test_spam_directory, test_ham_directory, dictionary, spam_prior_probability) 

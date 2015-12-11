import csv #reading text files
import sys
import numpy as np #multi-dimensional matrices
import matplotlib.pyplot as plt #plots
import unittest
import time

	#todo capitalization
	#todo rb

#global constants
deletionCost = 1
insertionCost = 1
substitutionCost = 1

def LevenshteinDistance(s, t, deletionCost, insertionCost, substitutionCost):
	#for all i and j, d[i,j] will hold the LevenshteinDistance between the first i characters of s and the first j characters of t
	#Note: standard approach is delection cost = insertion cost =  substitution cost = 1

	s = s.lower()
	t = t.lower()

	m = len(s)
	n = len(t)
	d = np.zeros((m+1, n+1)) # indices [0,0] to [m,n]
	
	#initialize array
	for i in range(0,m+1):
		d[i,0] = i*deletionCost

	for j in range(0,n+1):
		d[0,j] = j*insertionCost	

	for j in range (1,n+1):
		for i in range(1, m+1):

			if s[i-1] == t[j-1]:
				d[i,j] = d[i-1,j-1] #no operation cost, because they match
			else:
				d[i,j] = min(d[i-1,j] + deletionCost, d[i,j-1] + insertionCost, d[i-1, j-1] + substitutionCost)

	#print "Levenshtein Distance is: " + str(d[m,n])
	return d[m,n]

def find_closest_word (string1, dictionary):
	#write code to do this, call LD, and return a string (the closest word)
	
	closest_word = ""
	closest_word_LD = float("inf")

	for word in dictionary:

		ld = LevenshteinDistance(string1, word,deletionCost,insertionCost,substitutionCost)
		if ld < closest_word_LD:
			closest_word = word
			closest_word_LD = ld

	print "closest to \"" + string1 + "\": \"" + closest_word + "\" (" + str(closest_word_LD) + " LD score)"
	return closest_word;

#takes 3 lists of strings
def measure_error(typos, truewords, dictionary):
	#find whether corrected typo using the dictionary matches the true word
		#1 if it doesn't, 0 if it does. Count them all up and return a real value
		#between 0 and 1 representing the error_rate

	#initialize counts
	errors = 0.0
	total_tries = 0.0

	for i in range(0, len(typos)):
		found_word = find_closest_word(typos[i],dictionary)
		correct_word = truewords[i]

		#increment as necessary
		if (found_word != correct_word):
			errors = errors + 1
		
		total_tries = total_tries + 1

	error_rate = (errors / total_tries)

	return error_rate

# typos = []
# truewords = []
# dictionary = []
#print measure_error(typos, truewords, dictionary) 
#print ''

#Problem 3 Parameter Picking

def run_measure_error_over_dataSet():

	dictionary = [] #initialize Dictionary list
	dictFileName = "3esl.txt"
	masterFile = "wikipediatypoclean_small.txt"

	with open(dictFileName, 'rb') as csvfile:
		dictFile = csv.reader(csvfile, dialect= 'excel-tab')

		for row in dictFile:
			word = "".join(row)
			dictionary.append(word)

	with open(masterFile, 'rb') as csvfile:
		incorrectFile = csv.reader(csvfile, dialect= 'excel-tab')

		typos = []
		truewords = []

		for row in incorrectFile:
			incorrect_word = row[0]
			correct_word = row[1]

			typos.append(incorrect_word)
			truewords.append(correct_word)

	print measure_error(typos, truewords, dictionary)

#start = time.time()
#run_measure_error_over_dataSet()
#print "Time it took run function: " + str((time.time() - start))

#takes in two strings of length 1, returns qwerty distance
def qwerty_substitution_cost(s, t):

	dict = {"z": (0,0), "x": (0,1), "c": (0,2), "v": (0,3), "b": (0,4), "n": (0,5), "m": (0,6),
			"a": (1,0), "s": (1,1), "d": (1,2), "f": (1,3), "g": (1,4), "h": (1,5), "j": (1,6), "k": (1,7), "l": (1,8),
			"q": (2,0), "w": (2,1), "e": (2,2), "r": (2,3), "t": (2,4), "y": (2,5), "u": (2,6), "i": (2,7), "o": (2,8), "p": (2,9)}
	
	#check whether character is between a-z; if not, return subs_cost of infinity
	if not (s.isalpha() & t.isalpha()):
		return float("inf")

	first_letter = dict[s.lower()]
	second_letter = dict[t.lower()]
	
	dist_x = first_letter[0] - second_letter[0] 
	dist_y = first_letter[1] - second_letter[1]

	cost = abs(dist_x) + abs(dist_y)

	return cost

#s = "Z"
#t = "t"
 
#print "qwerty distance between keys " + s + " & " + t + " is: " + str(qwerty_substitution_cost(s,t))
#print "qwerty distance between keys " + t + " & " + s + " is: " + str(qwerty_substitution_cost(t,s))

def qwerty_levenshtein_distance(s, t, deletionCost, insertionCost):

	s = s.lower()
	t = t.lower()

	m = len(s)
	n = len(t)
	d = np.zeros((m+1, n+1)) # indices [0,0] to [m,n]
	
	#initialize array
	for i in range(0,m+1):
		d[i,0] = i*deletionCost

	for j in range(0,n+1):
		d[0,j] = j*insertionCost	

	for j in range (1,n+1):
		for i in range(1, m+1):

			if s[i-1] == t[j-1]:
				d[i,j] = d[i-1,j-1] #no operation cost, because they match
			else:

				sub_cost  = qwerty_substitution_cost(s[i-j],t[j-1])
				#print "sub cost between "+ s[i-j] + " & " +t[j-1] + ": " + str(sub_cost)
				d[i,j] = min(d[i-1,j] + deletionCost, d[i,j-1] + insertionCost, d[i-1, j-1] + sub_cost)
				#print "distance between "+ s[i-j] + " & " +t[j-1] + ": " + str(d[i,j])

	#print "qwerty Levenshtein Distance is: " + str(d[m,n])
	return d[m,n]

string1 = "hello"
string2 = "bello"
string3 = "qello"

#print "qwerty distance between " + string1 + " & " +string2 + " is: " + str(qwerty_levenshtein_distance(string1, string2,deletionCost,insertionCost))
#print "qwerty distance between " + string1 + " & " +string3 + " is: " + str(qwerty_levenshtein_distance(string3, string2,deletionCost,insertionCost)) 


print ""
def main():

	numArgs = len (sys.argv)
	correctNumArgs = 3
	
	if numArgs < correctNumArgs:
		print "Error: Incorrect number of args-- \n Usage: spellcheck.py <ToBeSpellCheckedFileName> <DictionaryFileName>"
		return

	ToBeSpellCheckedFile = sys.argv[1]
	dictFileName = sys.argv[2]

	dictionary = [] #initialize Dictionary list

	with open(dictFileName, 'rb') as csvfile:
		dictFile = csv.reader(csvfile, dialect= 'excel-tab')

		for row in dictFile:
			word = "".join(row)
			dictionary.append(word)

	with open(ToBeSpellCheckedFile, 'rb') as csvfile:
		incorrectFile = csv.reader(csvfile, dialect= 'excel-tab')

		incorrectWords = []

		for row in incorrectFile:
			word = "".join(row)
			incorrectWords.append(word)

	correctedFile = open("corrected.txt", 'w+')

	for word in incorrectWords:
		correctWord = find_closest_word(word, dictionary)
		correctedFile.write(correctWord + "\n")

	correctedFile.close()

if __name__ == "__main__":
	main()
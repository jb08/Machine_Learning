# Starter code for uesr-based collaborative filtering
# Complete the function user_based_cf below. Do not change it arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
from collections import Counter
import scipy.spatial 
import sys
import csv
import gc
import time
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

def user_based_cf(datafile, userid, movieid, distance, k, iFlag):
    '''
    build user-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set; 
    <distance> bool 0 pearson, 1 manhattan
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For user-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Jason Brown
    '''

    ratings_db = build_db(datafile)

    num_users = ratings_db.shape[0]
    num_movies = ratings_db.shape[1]

    gc.disable()

    d_from_user = []

    #iterate through all movies to determine closest movie
    for i in range(0,num_users):
        
        dist = float("inf")
        pair = (i, dist)

        if (i != userid -1):
            
            userid_row = ratings_db[userid-1]
            userid_row_excl_movieid = np.delete(userid_row, movieid-1)

            i_row = ratings_db[i]
            i_row_excl_movieid = np.delete(i_row, movieid-1)
            
            if(distance==0):
                dist = scipy.stats.pearsonr(userid_row_excl_movieid, i_row_excl_movieid)
                dist = dist[0]
                dist = abs(1-dist)

            else:
                dist = scipy.spatial.distance.cityblock(userid_row_excl_movieid, i_row_excl_movieid)
                #print "dist between movie and movie i: " + str(i) + " is: " + str(dist)
            pair = (i, dist)

        d_from_user.append(pair)

    gc.enable()

    nearest_users = sorted(d_from_user, key= lambda x:x[1])
    #print "distance from userid to other users (userid, distance): " + str(nearest_users)

    nearest_user = (nearest_users[0])[0]

    nearest_k_movie_ratings = []

    i = 0
    while i<k:

        nearest_pair = nearest_users.pop(0)
        nearest_user = nearest_pair[0]

        rating = ratings_db[nearest_user][movieid-1]
        
        if(iFlag == 1 or rating != 0):

            nearest_k_movie_ratings.append(rating)
            i +=1

    #print "k nearest movies:" + str(nearest_k_movie_ratings)

    data = Counter(nearest_k_movie_ratings)
    #data.most_common()   # Returns all unique items and their counts
    predictedRating = (data.most_common(1)[0])[0]  # Returns the highest occurring item

    #predictedRating = -1
    trueRating = ratings_db[userid-1, movieid-1]
  
    return trueRating, predictedRating

def build_db(datafile,num_users = 943, num_movies = 1682):

    ratings_data_raw = np.loadtxt(datafile, np.integer)

    ratings_db = np.zeros((num_users, num_movies), dtype=np.integer)

    for i in range(0,ratings_data_raw.shape[0]):
        user_id = ratings_data_raw[i][0]
        item_id = ratings_data_raw[i][1]
        rating = ratings_data_raw[i][2]
  
        ratings_db[user_id-1][item_id-1] = rating

    #print "loading data completed"

    return ratings_db

def build_charts(datafile, num_users, num_movies, userpairs = False):

    ratings_db = build_db(datafile, num_users, num_movies)
    
    reviews_in_common = np.zeros((num_users*num_users))
    index = 0

    #num_users = 500

    if (userpairs):

        gc.disable()

        for i in range(0,num_users):

            for j in range(i+1,num_users):
                
                one_pair = 0

                for k in range(0, num_movies):

                    if (ratings_db[i][k] >0 and ratings_db[j][k] > 0):
                        one_pair +=1

                if(one_pair>0):
                    reviews_in_common[index] = one_pair
                    index +=1

        #print reviews_in_common.size
        reviews_in_common = reviews_in_common[:i]
        #print reviews_in_common.size

        gc.enable()

        reviews_median = np.median(reviews_in_common)
        print "reviews in common median: " + str(reviews_median)

        reviews_mean = np.mean(reviews_in_common)
        print "reviews in common mean: " + str(reviews_mean)

        #print reviews_in_common
        reviews_in_common = filter(lambda a: a != 0, reviews_in_common)

        #print "filtered: " + str(reviews_in_common)

        #num_pairs = np.zeros((50), dtype=np.integer)
        # for num_reviews in reviews_in_common:
        #     num_pairs[num_reviews] +=1

        #print num_pairs

        #the histogram of the data
        plt.hist(reviews_in_common, facecolor='g', alpha=0.75)

        plt.xlabel('Reviews in Common')
        plt.ylabel('Number of User Pairs')
        plt.title('Histogram of Reviews in Common')
        #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        #plt.axis([0, 50 , 0, np.max(reviews_in_common)*2])
        plt.grid(True)
        plt.show()

    gc.disable()

    movie_reviews = []

    for k in range(0,num_movies):
        one_movie = 0

        for i in range(0,num_users):
             if (ratings_db[i][k] >0):
                    one_movie += 1

        movie_reviews.append(one_movie)

    max_movieid = -1
    max_reviews = float("-inf")

    min_movieid = -1
    min_reviews = float("inf")

    for k in range (0,num_movies):
        if movie_reviews[k]>max_reviews:
            max_movieid = k
            max_reviews = movie_reviews[k]
        
        if movie_reviews[k]< min_reviews:
                min_movieid = k
                min_reviews = movie_reviews[k]

    gc.enable()

    print "movie " + str(max_movieid+1) + " had max reviews: " + str(max_reviews)
    print "movie " + str(min_movieid+1) + " had min reviews: " + str(min_reviews)

    #print "movie reviews unsorted: " + str(movie_reviews)
    movie_reviews.sort( reverse= True)
    #print "movie reviews sorted: " + str(movie_reviews)

    t1 = np.arange(1,num_movies+1)
    plt.plot(t1, movie_reviews, 'b--')
    plt.title("Movie Reviews (Zipf's law?)")
    plt.ylabel("Movie Reviews")
    plt.xlabel("Number of reviews")
    plt.axis([0, num_movies+2, -1, max_reviews+2])
    plt.grid(True)
    plt.show()

    return ratings_db

def main():
    start_time = time.time()

    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])

    #build_charts(datafile, 943,1682)
    print("--- %s seconds ---" % (time.time() - start_time))

    trueRating, predictedRating = user_based_cf(datafile, userid, movieid, distance, k, i)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)

if __name__ == "__main__":
    main()
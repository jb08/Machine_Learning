# Starter code for item-based collaborative filtering
# Complete the function item_based_cf below. Do not change its name, arguments and return variables. 
# Do not change main() function, 

# import modules you need here.
import scipy.spatial 
import sys
import gc
import scipy.stats
import numpy as np
from collections import Counter

def build_db(datafile,num_users, num_movies):

    ratings_data_raw = np.loadtxt(datafile, np.integer)

    ratings_db = np.zeros((num_users, num_movies), dtype=np.integer)

    for i in range(0,ratings_data_raw.shape[0]):
        user_id = ratings_data_raw[i][0]
        item_id = ratings_data_raw[i][1]
        rating = ratings_data_raw[i][2]
    
        ratings_db[user_id-1][item_id-1] = rating

    return ratings_db

def item_based_cf(datafile, userid, movieid, distance, k, iFlag):
    '''
    build item-based collaborative filter that predicts the rating 
    of a user for a movie.
    This function returns the predicted rating and its actual rating.
    
    Parameters
    ----------
    <datafile> - a fully specified path to a file formatted like the MovieLens100K data file u.data 
    <userid> - a userId in the MovieLens100K data
    <movieid> - a movieID in the MovieLens 100K data set
    <k> - The number of nearest neighbors to consider
    <iFlag> - A Boolean value. If set to 0 for user-based collaborative filtering, 
    only users that have actual (ie non-0) ratings for the movie are considered in your top K. 
    For item-based, use only movies that have actual ratings by the user in your top K. 
    If set to 1, simply use the top K regardless of whether the top K contain actual or filled-in ratings.

    returns
    -------
    trueRating: <userid>'s actual rating for <movieid>
    predictedRating: <userid>'s rating predicted by collaborative filter for <movieid>


    AUTHOR: Jason Brown
    '''

    ratings_db = build_db(datafile, 943, 1682)
    #ratings_db = np.array([[5,2,2,4], [1,4,4,2], [2,4,3,2],[5,1,2, 3]])

    num_users = ratings_db.shape[0]
    num_movies = ratings_db.shape[1]

    gc.disable()

    d_from_movie = []

    #iterate through all movies to determine closest movie
    for i in range(0,num_movies):
        
        dist = float("inf")
        pair = (i, dist)

        if (i != movieid -1):
            
            movieid_col = ratings_db[:,[movieid-1]]
            movieid_col_excl_userid = np.delete(movieid_col, userid-1,0)

            #print movieid_col_excl_userid

            i_col = ratings_db[:,[i]]
            i_col_excl_userid = np.delete(i_col, userid-1, 0)

            #print i_col_excl_userid
            
            if(distance==0):
                dist = scipy.stats.pearsonr(movieid_col_excl_userid, i_col_excl_userid)
                dist = dist[0]
                dist = abs(1-dist)

            else:
                dist = scipy.spatial.distance.cityblock(movieid_col_excl_userid, i_col_excl_userid)
                #print "dist between movie and movie i: " + str(i) + " is: " + str(dist)
            pair = (i, dist)

        d_from_movie.append(pair)

    gc.enable()

    nearest_movies = sorted(d_from_movie, key= lambda x:x[1])
    #print "nearest movies to movieid: " + str(nearest_movies)

    nearest_movie = (nearest_movies[0])[0]

    predictedRating = ratings_db[userid-1][nearest_movie]
    
    nearest_k_movie_ratings = []

    i = 0
    while i<k:
        #print nearest_movies

        nearest_pair = nearest_movies.pop(0)
        nearest_movie = nearest_pair[0]

        rating = ratings_db[userid-1][nearest_movie]
        
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

def main():

    datafile = sys.argv[1]
    userid = int(sys.argv[2])
    movieid = int(sys.argv[3])
    distance = int(sys.argv[4])
    k = int(sys.argv[5])
    i = int(sys.argv[6])
    
    trueRating, predictedRating = item_based_cf(datafile, userid, movieid, distance, k, i)
    print 'userID:{} movieID:{} trueRating:{} predictedRating:{} distance:{} K:{} I:{}'\
    .format(userid, movieid, trueRating, predictedRating, distance, k, i)

if __name__ == "__main__":
    main()
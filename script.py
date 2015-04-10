import numpy as np
import pandas as pd
import csv

# Predict via the user-specific median.
# If the user has no data, use the global median.

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'predictions.csv'

# Load the training data.
print 'loading training data'
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = int(plays)

# Compute the global median and per-user median.
print 'computing global and user medians'
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))

# Read user-artist.csv
print 'reading user-artist.csv'
# user_art = pd.read_csv('user-artist.csv', index_col=0)

# Read artist-features.csv
print 'reading artist-features.csv'
art_feat = pd.read_csv('artist-features.csv', index_col=1)
art_feat['hotttnesss'] -= art_feat['hotttnesss'].mean()
art_feat['hotttnesss'] /= art_feat['hotttnesss'].std()
art_feat['familiarity'] -= art_feat['familiarity'].mean()
art_feat['familiarity'] /= art_feat['familiarity'].std()
art_feat['start'] -= art_feat['start'].mean()
art_feat['start'] /= art_feat['start'].std()


def compare(a1, a2):
    tot = 0
    if a1['genre'] != a2['genre']:
        tot += 2
    tot += (a1['hotttnesss'] - a2['hotttnesss']) ** 2.
    tot += (a1['familiarity'] - a2['familiarity']) ** 2.
    tot += (a1['start'] - a2['start']) ** 2.
    if a1['active'] != a2['active']:
        tot += 1
    return tot


# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])

        print 'first pass to compute distances'
        dists = []
        i = 0
        for row in test_csv:
            i += 1
            if i >= 11:
                break
            id     = row[0]
            user   = row[1]
            artist = row[2]

            # make predictions
            profile = train_data[user]
            tot = 0.
            for a in profile:
                tot += profile[a]*compare(art_feat.loc[a], art_feat.loc[artist])
            tot /= float(sum(profile.values()))
            dists.append(tot)
        
        print 'making distances zero-mean'
        mn = np.mean(dists)
        dists = [x - mn for x in dists]
        
        print 'writing out test predictions'
        i = 0
        for row in test_csv:
            i += 1
            if i >= 11:
                break
            id     = int(row[0])
            user   = row[1]
            artist = row[2]

            if user in user_medians:
                print id
                soln_csv.writerow([id, user_medians[user] - 1*dists[id-1]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])
                

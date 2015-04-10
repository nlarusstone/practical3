import numpy as np
import pandas as pd
import csv
from collections import Counter

from sklearn.utils import validation
validation.check_arrays = validation.check_array
from sklearn.ensemble import GradientBoostingRegressor as GBR

import time

N = 4154805

# model:  (# plays on user, artist) ~ user_sex + user_age + user_country
# note: model completely ignores artist and tastes

# Load the training data - profiles.csv
print 'loading profiles...'
profiles = pd.read_csv('profiles.csv', index_col=0)

print 'analyzing profiles...'
counter = Counter()
id_to_info = {}
for i, row in profiles.iterrows():
    id_to_info[row.name] = {'sex': row['sex'], 'age': row['age'], 'country': row['country']}
    counter[row['country']] += 1

countries = dict(counter.most_common(20)).keys() + ['Other']

# Load targets - play counts in train.csv
print 'loading train...'
train = pd.read_csv('train.csv', nrows=N)

print 'treating train...'
preds = []
targets = []
for i, row in train.iterrows():
    user = row['user']
    user_prof = id_to_info[user]
    sex = user_prof['sex'] if user_prof['sex'] in ['m', 'f'] else np.random.choice(['m', 'f'])
    sex = 1 if sex == 'm' else 0
    age = int(user_prof['age']) if not np.isnan(user_prof['age']) else np.mean(profiles.age)
    age = min(50, age)
    age = max(12, age)
    country = user_prof['country']
    newrow = [sex, age]
    other = True
    for c in countries:
        if c == country:
            newrow.append(1)
            other = False
        else:
            newrow.append(0)
    if other:
        newrow.append(1)
    else:
        newrow.append(0)
    preds.append(newrow)
    targets.append(int(row['plays']))

preds = np.array(preds)
targets = np.array(targets)

# fit model
print 'fitting model...'
gbr = GBR(loss='huber', n_estimators=100)
gbr.fit(preds, targets)

# Load test data - test.csv
print 'loading test...'
raw_test = pd.read_csv('test.csv', nrows=N)

print 'treating test...'
test = []
for i, row in raw_test.iterrows():
    user = row['user']
    user_prof = id_to_info[user]
    sex = user_prof['sex'] if user_prof['sex'] in ['m', 'f'] else np.random.choice(['m', 'f'])
    sex = 1 if sex == 'm' else 0
    age = int(user_prof['age']) if not np.isnan(user_prof['age']) else np.mean(profiles.age)
    age = min(50, age)
    age = max(12, age)
    country = user_prof['country']
    newrow = [sex, age]
    other = True
    for c in countries:
        if c == country:
            newrow.append(1)
            other = False
        else:
            newrow.append(0)
    if other:
        newrow.append(1)
    else:
        newrow.append(0)
    test.append(newrow)

test = np.array(test)

# predict on test data
print 'making predictions...'
sols = gbr.predict(test)

# Write out test solutions.
with open('demo_preds.csv', 'w') as fh:
    sol_csv = csv.writer(fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    sol_csv.writerow(['Id', 'plays'])
    for i in xrange(N):
        sol_csv.writerow([i+1, max(0, sols[i])])


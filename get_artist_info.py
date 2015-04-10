from pyechonest import artist
from pyechonest.util import EchoNestAPIError
import pandas as pd
import numpy as np
import time

df = pd.read_csv('artists.csv')

data = []

for i, row in df.iterrows():
    newrow = [row['artist'], row['name']]
    try:
        a = artist.Artist('musicbrainz:artist:' + row['artist'], buckets=['hotttnesss', 'familiarity', 'terms', 'years_active'])
        if a.terms:
            newrow.append(a.terms[0]['name']) # genre
        else:
            newrow.append('not a genre')
        newrow.append(a.hotttnesss)
        newrow.append(a.familiarity)
        if a.years_active:
            newrow.append(int(a.years_active[0]['start']))
            newrow.append(0 if 'end' in a.years_active[len(a.years_active)-1] else 1)
        else:
            newrow.extend([np.nan, np.nan])
        data.append(newrow)
    except EchoNestAPIError as e:
        if e.http_status == 429:
            print 'sleeping...'
            time.sleep(60)
            print 'and we\'re back'
            try:
                a = artist.Artist('musicbrainz:artist:' + row['artist'], buckets=['hotttnesss', 'familiarity', 'terms', 'years_active'])
                if a.terms:
                    newrow.append(a.terms[0]['name']) # genre
                else:
                    newrow.append('not a genre')
                newrow.append(a.hotttnesss)
                newrow.append(a.familiarity)
                if a.years_active:
                    newrow.append(int(a.years_active[0]['start']))
                    newrow.append(0 if 'end' in a.years_active[len(a.years_active)-1] else 1)
                else:
                    newrow.extend([np.nan, np.nan])
                data.append(newrow)
            except EchoNestAPIError as e:
                if e.http_status == 200:
                    newrow.append('not a genre')
                    for i in xrange(4):
                        newrow.append(np.nan)
                    data.append(newrow)
        elif e.http_status == 200:
            newrow.append('not a genre')
            for i in xrange(4):
                newrow.append(np.nan)
            data.append(newrow)

new_df = pd.DataFrame(data, columns=['artistID', 'artistName', 'genre', 'hotttnesss', 'familiarity', 'start', 'active'])
new_df.fillna({col: new_df[col].mean() for col in new_df}, inplace=True)

new_df.to_csv('artist-features.csv')

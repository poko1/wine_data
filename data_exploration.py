import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import label encoder
from sklearn import preprocessing
# label_encoder object knows 
# how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(121)
df = pd.read_csv('final_data/final.csv')
ax1.hist(df['points'],bins=50)
ax1.set_yscale('log')
#df['price'].hist(bins=50)
#plt.savefig('price_dist.png')
#sns_fig = sns.histplot(df['price'], kde = False, color = 'b')
#fig = sns_fig.get_figure()
fig.savefig("score_dist.jpg") 

# Encode labels in column 'species'.
#df['country']= label_encoder.fit_transform(df['country'])
#df['province']= label_encoder.fit_transform(df['province'])
#df['variety']= label_encoder.fit_transform(df['variety'])
#df['winery']= label_encoder.fit_transform(df['winery'])
#df['designation']= label_encoder.fit_transform(df['designation'])
#df['taster_name']= label_encoder.fit_transform(df['taster_name'])
#df = df[['variety', 'points']]
#plt.figure(figsize=(12, 6))
#sns_fig = sns.heatmap(df.corr(),
#            cmap = 'BrBG',
#            fmt = '.2f',
#            linewidths = 2,
#            annot = True)

#fig = sns_fig.get_figure()
#fig.savefig("out-.jpg") 
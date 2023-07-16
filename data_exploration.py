#import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Import label encoder
#from sklearn import preprocessing
# label_encoder object knows 
# how to understand word labels.
#label_encoder = preprocessing.LabelEncoder()
#fig = plt.figure(figsize=(12,5))
#ax1 = fig.add_subplot(121)
df = pd.read_csv('data/winery.csv')
df = df.sample(n = 100)

import spacy
nlp = spacy.load("en_core_web_lg")
#nlp = spacy.load("en_core_web_md")
import statistics
s=[]

for i in df['description']:
    for j in df['description']:
        doc1 = nlp(i)
        doc2 = nlp(j)
        if doc1 != doc2:
            s.append(doc1.similarity(doc2))

print(statistics.mean(s))
print(statistics.stdev(s))

#doc1 = nlp(u'This stunning wine is all about extreme elegance and subtlety. It opens with aromas of wild red berry, rose petal, sweet baking spice and a whiff aromatic herb while the focused palate delivers red cherry, crushed raspberry, licorice, pipe tobacco and white pepper. A firm backbone of polished tannins and vibrant acidity gives it an age-worthy structure. Its loaded with finesse and should develop even more complexity over time. Drink 2018 to 2033.') #h 595
#doc2 = nlp(u'From a year in which only single quinta vintage Ports were produced, this is a hugely concentrated toffee and dried fruit wine. It is rich with sweetness and a dense texture that is balancing complex flavors against the drier wood-aging flavors. It has been bottled just at the right stage of its development, rich and dense. Drink this beautiful wine now.') #h 240
#doc3 = nlp(u'A concentrated wine that is the epitome of dark, dense Malbec. Spice, tense acidity, black fruits, dark plums and licorice are all present in this impressive, dense wine. The power in this dense wine is very evident, expressed right up front. It should age, so drink from 2017.') #l 35
#doc4 = nlp(u'Aromas of dried herbs intertwine with dill, vanilla and other barrel spices on this blend. The cranberry flavors have a tart zing, with slightly drying tannins providing the frame.') #l 19


#print(doc1.similarity(doc2)) #high high #0.9557891609482467
#print(doc2.similarity(doc3)) #high low #0.9672694583724468
#print(doc3.similarity(doc4)) #low low #0.881343817467576

#ax1.hist(df['points'],bins=50)
#ax1.set_yscale('log')
#df['price'].hist(bins=50)
#plt.savefig('price_dist.png')
#sns_fig = sns.histplot(df['price'], kde = False, color = 'b')
#fig = sns_fig.get_figure()
#fig.savefig("score_dist.jpg") 

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
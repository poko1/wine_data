from pathlib import Path
import os
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def process(file):
    df = pd.read_csv(file)
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    # numeric null value handling by replacing with median 
    avg_price=df['price'].median() #as normal distribution
    index_price=df[df['price'].isna()==True].index
    for i in index_price:
        df.loc[i,'price']=avg_price

    # text normalization and preprocessing
    lemmatiser = WordNetLemmatizer()
    for i in df.loc[:,'description'].index:
        final_out=""
        sent=df.loc[i,'description']
        #punctuation check
        sentence1=sent.translate(str.maketrans('', '', string.punctuation))
        sentence=''.join(sentence1)
        #lower case conversion
        temp1=sentence.lower()
        #splitting
        temp=temp1.split()
        #lemmetizing the sentence and stopwords removal
        for word in temp:
            if word not in stopwords.words('english'): #stopwords removal
                b=lemmatiser.lemmatize(word, pos="v")        
                final_out=final_out+b+' '
                
        df.loc[i,'description']=final_out 
        #break


    return df


#"""
if __name__ == '__main__':
    # path to the new file
    local_path = Path('data')
    # get all files in local path
    files = os.listdir(local_path)
    files.sort()  # sorting just for caution
    
    # get the latest file and process it
    file = files[-1]
    print(file)
    df = process(Path('data',file))
    print(df['description'][0])
    print(df['description'][9])
    
    #save processed it
    processed_path = Path('processed_data')
    df.to_csv(processed_path / file, index=False)
    
    #merge all the processed data and save to outputs
    file_path = Path('processed_data')
    #list all the files from the directory
    file_list = os.listdir(file_path)
    print(file_list)
    outputs_path = Path('final_data')
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(Path(processed_path,f)) for f in file_list ])
    #export to csv
    combined_csv.to_csv( outputs_path /"final.csv", index=False, encoding='utf-8-sig')
#"""
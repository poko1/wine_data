import logging
from pathlib import Path
import os
import pandas as pd
from data_processing import process
from model_training import train_test_model, vectorizing
from model_loading import predict_user_input
from sklearn.model_selection import train_test_split


if __name__ == '__main__':

    # get all files from data folder and only process the latest
    logging.info('Data processing pipeline')
    local_path = Path('data')
    files = os.listdir(local_path)
    files.sort() 
    file = files[-1]
    # df = process(Path('data',file))
    processed_path = Path('processed_data')
    # df.to_csv(processed_path / file, index=False)
    # merge all the processed data and save to final_data directory
    file_path = Path('processed_data')
    file_list = os.listdir(file_path)
    outputs_path = Path('final_data')
    combined_csv = pd.concat([pd.read_csv(Path(processed_path,f)) for f in file_list ])
    combined_csv.to_csv( outputs_path /"final.csv", index=False, encoding='utf-8-sig')


    # train and test model using the final version of the data
    logging.info('Model training and vadilation pipeline')
    df = pd.read_csv(Path(outputs_path,"final.csv"))
    X = df['description']
    y = df[['points','price']]
    X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=104,test_size=0.25, shuffle=True)
    X_train, X_test, y_train, y_test = vectorizing(X_train, X_test, y_train, y_test)
    #train_test_model(X_train, X_test, y_train, y_test)

    # load the latest model and predict a new review from user
    #new_test = input()
    folder = "model"
    modelname = "finalized_model_all.sav"
    vectorname = "vectorizer_all.pickle"
    print("Enter description:")
    new_test = []
    str = input()
    new_test.append(str)
    result = predict_user_input(new_test, folder, modelname, vectorname)
    print(result)
    std_score_low= df['points'].quantile(0.1)  # 10th percentile
    std_score_high= df['points'].quantile(0.9)  # 90th percentile
    std_price_low= df['price'].quantile(0.1)  # 10th percentile
    std_price_high= df['price'].quantile(0.9)  # 90th percentile
    #print(std_score_low)
    #print(std_score_high)
    #print(std_price_low)
    #print(std_price_high)
    # calculate cosine similarity between this sentence and 100 random training sentences
    df = df.sample(n = 100)
    s = []
    for i in df['description']:
        doc1 = nlp(i)
        doc2 = nlp(str)
        s.append(doc1.similarity(doc2))

    print(statistics.mean(s))
    print(statistics.stdev(s))
    if(result[0][0]<std_score_low or result[0][0]>std_score_high or result[0][1]<std_price_low or result[0][1]>std_price_high or statistics.mean(s)<0.91 or statistics.mean(s)>0.95):
       logging.info("Warning: model prediction crossed threshold")
    






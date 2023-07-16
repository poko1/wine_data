# load the model from disk
import pickle
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from fastapi import FastAPI
from joblib import load
import joblib


def predict_user_input(new_test, folder, modelname, vectorname):
    loaded_model = pickle.load(open(Path(folder, modelname), 'rb'))
    vectorizer = pickle.load(open(Path(folder, vectorname), 'rb'))
    temp1=new_test[0].lower()
    temp=temp1[0].split()
    final_out = ''
    #lemmetizing the sentence and stopwords removal
    lemmatiser = WordNetLemmatizer()
    for word in temp:
        if word not in stopwords.words('english'): #stopwords removal
            b=lemmatiser.lemmatize(word, pos="v")        
            final_out=final_out+b+' '
    new_test = [final_out]
    new_test = vectorizer.transform(new_test)
    result = loaded_model.predict(new_test)
    #print(result)
    #print(loaded_model.predict(new_test).std())
    return result

def classify(classifier, text, vectorname):
    vectorizer = pickle.load(open(Path(folder, vectorname), 'rb'))
    new_test = []
    new_test.append(new_test)
    new_test = vectorizer.transform(text)
    prediction = classifier.predict([[new_test]])
    return {'label': prediction}



#if __name__ == '__main__':
    # load the model
folder = "model"
modelname = "finalized_model.sav"
vectorname = "vectorizer.pickle"
classifier = joblib.load(Path(folder, modelname))
app = FastAPI()

@app.get('/')
def home():
    return {'message': 'Wine review data to gauge score and price'}

@app.post('/tinatic_prediction')
def prediction(text: str):
    return classify(classifier, text, vectorname)
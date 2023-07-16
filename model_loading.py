# load the model from disk
import pickle
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def predict_user_input(new_test, folder, modelname, vectorname):
    loaded_model = pickle.load(open(Path(folder, modelname), 'rb'))
    vectorizer = pickle.load(open(Path(folder, vectorname), 'rb'))
    temp1=new_test[0].lower()
    temp=temp1.split()
    final_out = ''
    #lemmetizing the sentence and stopwords removal
    lemmatiser = WordNetLemmatizer()
    for word in temp:
        if word not in stopwords.words('english'): #stopwords removal
            b=lemmatiser.lemmatize(word, pos="v")        
            final_out=final_out+b+' '
    print(final_out)
    new_test = [final_out]
    new_test = vectorizer.transform(new_test)
    result = loaded_model.predict(new_test.toarray())
    print(result)
    print(loaded_model.predict(new_test).std())
    return result



#"""
if __name__ == '__main__':
    # load the model
    folder = "model"
    modelname = "finalized_model_all.sav"
    vectorname = "vectorizer_all.pickle"
    new_test = ["smooth operator sweet aromas cotton candy caramel bake black plum palate exceedingly flush yet refine meaty flavor profile entail roast berry fruit herbs finish sweet easy note caramel toast drink 2013"]
    predict_user_input(new_test, folder, modelname, vectorname)
#"""

# https://medium.com/codex/house-price-prediction-with-machine-learning-in-python-cf9df744f7ff
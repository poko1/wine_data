FROM python:latest
# 
WORKDIR /code
# 
COPY requirements.txt /code/requirements.txt
# 
RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt  
#
COPY model/vectorizer.pickle /code/vectorizer.pickle
COPY model/finalized_model_all.sav /code/finalized_model_all.sav
#
COPY main.py /code/main.py
#
CMD ["python", "main.py"]
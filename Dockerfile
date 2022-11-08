#FROM jupyter/datascience-notebook
FROM python:3
WORKDIR /app
EXPOSE 5000 
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app 
RUN [ "python3", "-c", "import nltk; nltk.download('punkt', download_dir='/usr/local/nltk_data')" ]
CMD [ "python", "./1/english_classifier.py" ]

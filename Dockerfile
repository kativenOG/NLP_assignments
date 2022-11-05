FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN apt install python3-tk 
CMD [ "python", "./1/english_classifier.py" ]


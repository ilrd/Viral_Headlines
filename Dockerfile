FROM tensorflow/tensorflow:2.4.0-gpu

MAINTAINER Illia Herasymenko, illia.cgerasimenko@gmail.com

RUN mkdir -p /app/ && mkdir -p /root/.cache/huggingface

COPY huggingface /root/.cache/huggingface

WORKDIR /app/

COPY docker_requirements.txt .

RUN pip install --upgrade pip && pip install -r docker_requirements.txt && python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords

COPY . .

CMD ["python", "src/modelinghon", "src/modeling/tr/training_testing.py"]
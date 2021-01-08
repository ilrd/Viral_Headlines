FROM tensorflow/tensorflow:2.4.0-gpu

MAINTAINER Illia Herasymenko, illia.cgerasimenko@gmail.com

RUN mkdir -p /app/ && mkdir -p /root/.cache/huggingface

COPY huggingface /root/.cache/huggingface

WORKDIR /app/

COPY docker_requirements.txt .

RUN pip install --upgrade pip && pip install -r docker_requirements.txt && python -m nltk.downloader punkt averaged_perceptron_tagger wordnet stopwords

COPY . .

CMD ["python", "src/modelinghon", "src/modeling/tr/training_testing.py"]

# import src.modeling.training_testing
# apt-get -y install python3-tk
# docker run -it --rm --gpus all --env="DISPLAY" --volume="$PWD":/app --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" viral_headlines:2.4.0
# docker run -it --rm --gpus all --env="DISPLAY" --net=host -e DISPLAY --volume="$PWD":/app --volume="/etc/group:/etc/group:ro" --volume="/etc/passwd:/etc/passwd:ro" --volume="/etc/shadow:/etc/shadow:ro" --volume="/etc/sudoers.d:/etc/sudoers.d:ro" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" viral_headlines:2.4.0 bash
# docker run -it --rm --gpus all --net=host -e DISPLAY viral_headlines:2.4.0 bash
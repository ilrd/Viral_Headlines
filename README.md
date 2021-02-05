# Viral Headlines


The project is made for creative YouTubers that want to have the most
audience-attractive headlines possible or to make their headlines serve any
other goal they want (to get many likes but few dislikes for example).


In this project, I built a Neural Network that analyses videos’ headlines on their
user-attractiveness (predicts whether many people will click on the video),
positive impression (predicts whether many people will like the video),
negative impression (predicts whether many people will dislike the video), and
sentiment of the headline (negative or positive). The project focuses mainly on
political videos, but with the potential to expand to other topics.


I built the Neural Net using Keras framework which you can see in model.py
file. Transformer model DistilBert extracts sentiments from the headlines. To
get the data for Neural Net training, I scraped 80k+ youtube videos from the
most popular political YouTube channels using my own youtube scraper that
you can find in youtube_scraper.py.


From other files, there are nlp_preprocessing.py in which a class for
text-related preprocessing is declared, preprocessing.py in which all the actual
preprocessing happens. news.csv file contains the preprocessed data that is
later used in training_testing.py file for model training and evaluation.


To run the project it is best to:
- Have python 3.8
- Install all dependencies (run “pip install -r requirements.txt” from the root
folder of the project)
- Run “python src/modeling/training_testing.py” from the root folder and
follow the instructions in the console
  

I also deployed the project into the web using Flask framework on
http://illiaherasymenko.pythonanywhere.com/ , but unfortunately due to the
limited storage on the web host, I was unable to run Tensorflow there.
Anyway, you can run server.py and open http://localhost:5000/ in your
browser to see how the model works in the deployment. Also, you can build a
docker image of this project using the provided Dockerfile.

## Docker
To run the project with docker you can use the image I hosted on docker hub: https://hub.docker.com/repository/docker/illiaherasymenko/viral_headlines/general

From the console run:

docker pull illiaherasymenko/viral_headlines:final
docker run --rm -ti --gpus all illiaherasymenko/viral_headlines:final

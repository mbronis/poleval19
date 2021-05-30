# Submission for [PolEval 2019](http://2019.poleval.pl/) task 6-2

## Task info
Classify tweets into three categories:
* **non harmful**
* **cyberbullying** (harmful action is addressed towards a private person)
* **hate-speach** (harmful action is addressed towards a public person/entity/large group)

The train data (avaliable [here](http://2019.poleval.pl/task6/task_6-2.zip)) consist of 10.041 tweets, of which 253	(2.5%) are labeled as `cyberbullying` and 598	(6.0%) as `hate-speach`.
Test set ([here](http://2019.poleval.pl/task6/task6_test.zip)) consist of 1.000 tweets, with 25	(2.5%) `cyberbullying` labels and 
109	(10.9%) `hate-speach` labels.

For evaluation `microF` was chosen as the primary metric and `macroF` as a secondary one.

## Models
Proposition of problem solution is based on `bag-of-words` along with a range of classification algorithms (`svm`, `rigde`, `rf`, `gbm`).
Details of building and testing the models can be find in the `jupyter notebook`.  

For deployment blend of `RF` and `Ridge` is used. Chosen blend scores f1-macro: 0.874 and f1-micro: 0.4047.

## App with tweet tagging endpoint
Chosen model is served with ``Uvicorn`` based api that takes incoming tweet and responds with predicted tag.  

You can set up app locally with: ``uvicorn src.app:app --port 8100 --host 0.0.0.0 --reload``

## Docker
The app can be deployed as ``Docker`` container.  

In order to **build** an image run: ``docker build -t poleval19:base .``  
Now you can **run** dockerized tagging app with:  ``docker run -p 8100:8100 poleval19:base``

Building a docker image requires `pl-spacy-model` in `/language_models` folder. The file can be downloaded from [here](http://zil.ipipan.waw.pl/SpacyPL?action=AttachFile&do=get&target=pl_spacy_model-0.1.0.tar.gz)

## Unit tests
Suite of unit test is provided. Those can be used in `CI/CD` pipeline for automatic testing our app.
Right now only `preproc` module is covered. 

## Load tests
Dockerized app performance will be tested with `locust` package.  
I will simulate load coming from 100 simultaneous user, each making a request once per 3 seconds.   
Each request will be a random tweet selected from sample of train set.  




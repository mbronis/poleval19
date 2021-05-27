# Submission for [PolEval 2019](http://2019.poleval.pl/) task 6-2.

## Task info
Classify tweets into three categories:
* **non harmful**
* **cyberbullying** (harmful action is addressed towards a private person)
* **hate-speach** (harmful action is addressed towards a public person/entity/large group)

The train data (avaliable [here](http://2019.poleval.pl/task6/task_6-2.zip)) consist of 10.041 tweets, of which 253	(2.5%) are labeled as `cyberbullying` and 598	(6.0%) as `hate-speach`.
Test set ([here](http://2019.poleval.pl/task6/task6_test.zip)) consist of 1.000 tweets, with 25	(2.5%) `cyberbullying` labels and 
109	(10.9%) `hate-speach` labels.

For evaluation `microF` was chosen as the primary metric and `macroF` as a secondary one.

## App with tweet tagging endpoint
Chosen model is served with ``Uvicorn`` based api that can take incoming tweet and respond with predicted tag.  

You can set up app with: ``uvicorn src.app:app --port 8100 --host 0.0.0.0 --reload``

## Docker
The app can be deployed as ``Docker`` container.  

In order to **build** an image run: ``docker build -t poleval19:base .``  
Now you can **run** dockerized tagging app with:  ``docker run -p 8100:8100 poleval19:base``




import random
from locust import HttpUser, TaskSet, task, between
from src.utils import Config

TWEETS_PATH = Config()['DATA']['tests_text']
with open(TWEETS_PATH) as f:
    raw_tweets = f.readlines()


class TagTweet(TaskSet):
    @task
    def predict(self):
        tweet = random.choice(raw_tweets)
        request_body = {"text": tweet}
        self.client.post('/tag_tweet', json=request_body)


class TagTweetLoadTest(HttpUser):
    tasks = [TagTweet]
    host = 'http://0.0.0.0:8100'
    stop_timeout = 20
    wait_time = between(1, 5)

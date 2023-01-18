import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk
import tweepy
nltk.download('stopwords')

consumer_token = 'CONSUMER_TOKEN'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_secret = 'ACCESS_SECRET'

auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

df = pd.read_csv("tweet_emotions.csv")
df['content'] = df['content'].replace(
    to_replace=r'^https?:\/\/.*[\r\n]*', value='', regex=True)
df['content'] = df['content'].replace(
    to_replace=r'^@[a-zA-Z0-9]+', value='', regex=True)

vectorizer = TfidfVectorizer(
    sublinear_tf=True, min_df=5, max_df=0.8, norm='l2', ngram_range=(1, 3), stop_words='english')

X = vectorizer.fit_transform(df['content'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.01, random_state=13)

print("--------------------NAIVE BAYES--------------------")

nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_prediction = nb.predict(X_test)
print(accuracy_score(nb_prediction, y_test))
print(classification_report(nb_prediction, y_test))

print("--------------------SUPPORT VECTOR MACHINES--------------------")

svc = LinearSVC(max_iter=200000)
svc.fit(X_train, y_train)
svc_prediction = svc.predict(X_test)
print(accuracy_score(svc_prediction, y_test))
print(classification_report(svc_prediction, y_test))

print("--------------------RANDOM FOREST CLASSIFIER--------------------")
# povecanje ovih parametara dovodi do boljih rezultata
rf = RandomForestClassifier(n_estimators=100, max_depth=20)
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
print(accuracy_score(rf_prediction, y_test))
print(classification_report(rf_prediction, y_test))

print("--------------------LOADING TWEETS--------------------")

hashtag = 'serbia'
num_tweets = 100
tweets = tweepy.Cursor(api.search_tweets, hashtag,
                       lang='en', tweet_mode='extended').items(num_tweets)
list_tweets = [tweet.full_text for tweet in tweets]

df = pd.DataFrame(list_tweets, columns=["tweet"])
df['tweet'] = df['tweet'].replace(
    to_replace=r'https?:\/\/.*[\r\n]*/g', value='', regex=True)
df['tweet'] = df['tweet'].replace(
    to_replace=r'[#@][a-zA-Z0-9_:]+', value='', regex=True)

tweets_tfidf = vectorizer.transform(df['tweet'])
nb_tweets = nb.predict(tweets_tfidf)
svc_tweets = svc.predict(tweets_tfidf)
rf_tweets = rf.predict(tweets_tfidf)

for i, tweet in enumerate(df['tweet']):
    print("TWEET", i, ":")
    print(tweet)
    print("PREDICTIONS:")
    print("\tNAIVE BAYES:", nb_tweets[i])
    print("\tSVC:", svc_tweets[i])
    print("\tRANDOM FOREST:", rf_tweets[i])

df['svc_tweets'] = svc_tweets
df_pie = df['svc_tweets'].value_counts()
plt.title(f"{num_tweets} #{hashtag} Tweets by sentiment predicted with SVC model")
plt.pie(df_pie, labels=df_pie.index, autopct='%1.1f%%')
plt.show()

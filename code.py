from nltk.corpus import reviewText
from nltk.classify import RandomForestClassifier
from nltk.classify.util import accuracy as nltk_accuracy


def extract_features(words):
    return dict([(word, True) for word in words])

def assign_sentiment(text):

    features_pos = [(extract_features(reviews.words(fileids=[f])),'Positive') for f in fileids_pos]
    features_pos = [(extract_features(reviews.words(fileids=[f])),'Neutral') for f in fileids_n]
    features_neg = [(extract_features(reviews.words(fileids=[f])),'Negative') for f in fileids_neg]

    threshold = 0.8
    num_pos = int(threshold*len(features_pos))
    num_n = int(threshold*len(features_n))
    num_neg = int(threshold*len(features_neg))

    # creating training and testing data
    features_train = features_pos[:num_pos] + features_neg[:num_n] +features_neg[:num_neg]
    features_test = features_pos[num_pos:] + features_neg[:num_n] + features_neg[num_neg:]

    # training classifier 
    classifier =RandomForestClassifier.train(features_train)
    probabilities = classifier.prob_classify(extract_features(text.split()))
    # Pick the maximum value
    predicted_sentiment = probabilities.max()
    print("Predicted sentiment:", predicted_sentiment)

    return predicted_sentiment  
# SentimentAnalyzer('It was not that good.')
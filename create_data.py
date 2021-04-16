from nltk.sentiment import SentimentIntensityAnalyzer
import twint
import pandas as pd
import os
import nltk
import re
import emoji
nltk.download('all')
c = twint.Config()
c.Search = "depression"
c.Limit = 1000
c.Store_csv = True
c.Output = "./data/raw_depressed_data.csv"
c.Lang = 'en'
p = twint.Config()
p.Search = "the"
p.Limit = 1000
p.Store_csv = True
p.Output = "./data/raw_everyday_data.csv"
p.Lang = 'en'

# try:
#     os.remove("./data/raw_depressed_data.csv")
# except FileNotFoundError:
twint.run.Search(c)
twint.run.Search(p)

words = set(nltk.corpus.words.words())


def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+", "", tweet)  # Remove @ sign
    tweet = re.sub('[^a-zA-Z0-9]', ' ', tweet)  # remove non-english word
    tweet = re.sub('\s+', ' ', tweet)
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+",
                   "", tweet)  # Remove http links
    tweet = " ".join(tweet.split())
    # Remove Emojis
    tweet = ''.join(c for c in tweet if c not in emoji.UNICODE_EMOJI)
    # Remove hashtag sign but keep the text
    tweet = tweet.replace("#", "").replace("_", " ")
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
                     if w.lower() in words or not w.isalpha())
    return tweet


# def translate(x):
#     return TextBlob(x).translate(to="th")
print("\nImporting Data\n")
df = pd.read_csv("./data/raw_depressed_data.csv")
pos = pd.read_csv("./data/raw_everyday_data.csv")
print("\nDone Importing Data\n")
label = []
analysed = 0
print("\n Sentiment Analysing the tweets\n")
sia = SentimentIntensityAnalyzer()
for tweets in df['tweet']:
    if sia.polarity_scores(tweets)['pos'] >= 0.5:
        label.append(0)
    else:
        label.append(1)
    analysed = analysed + 1
    if analysed % 100 == 0:
        print(f"Analysed {analysed} so far...")
pos_label = []
for tweets in pos['tweet']:
    if sia.polarity_scores(tweets)['pos'] < 0.5:
        pos_label.append(0)
    else:
        pos_label.append(1)
    analysed = analysed + 1
    if analysed % 100 == 0:
        print(f"Analysed {analysed} so far...")

print("\n Finished Sentiment Analysing the tweets\n")
df2 = pd.DataFrame({'Tweets': df['tweet'], 'label': label})
pos2 = pd.DataFrame({'Tweets': pos['tweet'], 'label': pos_label})

print(df2)
is_pos = pos2['label'] == 0
pos2 = pos2[is_pos]
df2 = df2.append(pos2, ignore_index=True, sort=True)
df2['Tweets'] = df2['Tweets'].map(lambda x: cleaner(x))
df2.sort_values(by=['label'], ascending=False)
df2[df2.Tweets.map(lambda x: x.isascii())]
df2.dropna(inplace=True)

df2.to_csv("./data/cleaned_data.csv", index=False)
df3 = pd.read_csv("./data/cleaned_data.csv")
print(f"Depressed Tweets: {df2.label.value_counts()[1]}")
print(f"Positive Tweets: {df2.label.value_counts()[0]}")
print("\nStart Cleaning..\n")

for index, i in df3.iterrows():

    if df3.label.value_counts()[1] > df3.label.value_counts()[0] and i['label'] == 1:
        df3 = df3.drop(index, errors='ignore')
        # print(f"Index: {index}\n I: {i}")
    elif df3.label.value_counts()[1] < df3.label.value_counts()[0] and i['label'] == 0:
        df3 = df3.drop(index, errors='ignore')


print("\nDone Cleaning..\n")
print(f"Depressed Tweets: {df3.label.value_counts()[1]}")
print(f"Positive Tweets: {df3.label.value_counts()[0]}")

df3.to_csv("./data/data.csv", index=False)
os.remove("./data/raw_depressed_data.csv")
os.remove("./data/raw_everyday_data.csv")
os.remove("./data/cleaned_data.csv")

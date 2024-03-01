from newspaper import Article
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt

url = 'https://timesofindia.indiatimes.com/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

articles = []
for link in soup.find_all('a', href=True):
    article_url = link['href']
    if 'timesofindia' in article_url:
        try:
            article = Article(article_url, language='en')
            article.download()
            article.parse()
            articles.append(article.text)
        except:
            pass

# Print the extracted articles
for i, article in enumerate(articles[:20]):
    print(f"Article {i+1}:\n{article}\n")
    
print(articles)

import nltk
from nltk.corpus import stopwords
print(stopwords.words('english'))

import string
string.punctuation

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import PorterStemmer
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import emoji

# Remove emojis from articles
articles_no_emojis = [emoji.demojize(article) for article in articles]

# Print articles without emojis
for i, article in enumerate(articles_no_emojis):
    print(f"Article {i+1}: {article}")




# Function to transform text
def transform_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stopwords.words('english') and token not in punctuation]
    ps = PorterStemmer()
    tokens = [ps.stem(token) for token in tokens]
    return tokens

# Transform each article
transformed_articles = [transform_text(article) for article in articles]

# Extract unigrams and bigrams
all_unigrams = [token for article_tokens in transformed_articles for token in article_tokens]

# Create a word cloud for unigrams
unigram_wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(" ".join(all_unigrams))


# Plot the word clouds
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(unigram_wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Unigram Word Cloud")


from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
cv = CountVectorizer()
from transformers import pipeline


# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(articles).toarray()

# Load pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Predict sentiment for each article
sentiments = []

for article in articles:
    # Truncate the article to 512 tokens
    truncated_article = article[:512]
    
    # Predict sentiment for the truncated article
    sentiment = sentiment_analysis(truncated_article)[0]
    sentiments.append(sentiment)

# Print the sentiment for each article
for i, article in enumerate(articles):
    print(f"Article {i+1}: {article}")
    print(f"Sentiment: {sentiments[i]['label']} ({sentiments[i]['score']:.2f})")
    print()





import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# IMDb movie reviews URL
url = 'https://www.imdb.com/title/tt22170036/reviews/?ref_=tt_ql_2'

# Send a GET request to the IMDb reviews page
response = requests.get(url)

# Parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all review containers
review_containers = soup.find_all('div', class_='review-container')

# Load pre-trained sentiment analysis model
sentiment_analysis = pipeline("sentiment-analysis")

# Process each review
# Process each review
for review_container in review_containers:
    # Extract review text
    review_text = review_container.find('div', class_='text show-more__control').text.strip()
    
    # Truncate the review text to 512 tokens
    truncated_review_text = review_text[:512]
    
    # Perform sentiment analysis
    sentiment = sentiment_analysis(truncated_review_text)[0]
    
    # Print review text and sentiment
    print("Review:", review_text)
    print("Sentiment:", sentiment['label'], "(Score:", sentiment['score'], ")")
    print('-' * 50)


















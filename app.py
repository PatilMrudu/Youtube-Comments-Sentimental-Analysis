from flask import Flask, render_template, request
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/new-url', methods=['POST'])
def new_url_predict():
    comment = request.form['comment']
    processed_comment = preprocess_comment(comment)
    features = vectorizer.transform([processed_comment])
    prediction = model.predict(features)[0]
    sentiment = get_sentiment_label(prediction)
    return render_template('result.html', comment=comment, sentiment=sentiment)
@app.route('/predict', methods=['POST'])
def predict():
    comment = request.form['comment']
    processed_comment = preprocess_comment(comment)
    features = vectorizer.transform([processed_comment])
    prediction = model.predict(features)[0]
    sentiment = get_sentiment_label(prediction)
    return render_template('result.html', comment=comment, sentiment=sentiment)

def preprocess_comment(comment):
    # Comment preprocessing code here
    comment = comment.lower()
    comment = comment.translate(str.maketrans('', '', string.punctuation))
    comment = remove_stopwords(comment)
    comment = stem_words(comment)
    return comment

def remove_stopwords(comment):
    stopwords_english = set(stopwords.words('english'))
    comment_tokens = comment.split()
    comment = ' '.join([word for word in comment_tokens if word not in stopwords_english])
    return comment

def stem_words(comment):
    stemmer = PorterStemmer()
    comment_tokens = comment.split()
    comment = ' '.join([stemmer.stem(word) for word in comment_tokens])
    return comment

def get_sentiment_label(prediction):
    if prediction == 0:
        return 'negative '
    elif prediction == 1:
        return 'neutral '
    elif prediction == 2:
        return 'positive '
    else:
        return 'unknown'


if __name__ == '__main__':
    app.run(debug=True, port=5001)

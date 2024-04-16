from flask import Flask, render_template, request
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF vectorizer and model
tfidf_vectorizer = joblib.load(r"tfidf_vectorizer.pkl")
model = joblib.load(r"sentiment_model_lr.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        try:
            # Get the review from the form
            review = request.form['review']
            
            # Convert the review to lowercase and remove leading/trailing spaces
            review = review.lower().strip()
            
            # Vectorize the review using TF-IDF
            review_vectorized = tfidf_vectorizer.transform([review])
            
            # Perform sentiment analysis using the loaded model
            sentiment = model.predict(review_vectorized)[0]  # Predict sentiment

            print("Raw Sentiment Value:", sentiment)  # Debugging output
            
            # Map the sentiment to a human-readable label
            sentiment_label = 'Negative' if sentiment == 1 else 'Positive'
            
            print("Sentiment Label:", sentiment_label)  # Debugging output
            
            return render_template('results.html', review=review, sentiment=sentiment_label)
        except Exception as e:
            return render_template('error.html', error=str(e))
        
if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=5000)
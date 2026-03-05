import joblib
import numpy as np
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, '..', 'models')

# Load the trained model
model = joblib.load(os.path.join(models_dir, 'sentiment_model.pkl'))
print("✓ Model loaded successfully")

# Load the TF-IDF vectorizer
vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
print("✓ Vectorizer loaded successfully")

# Load class mapping
class_mapping = joblib.load(os.path.join(models_dir, 'class_mapping.pkl'))
print("✓ Class mapping loaded successfully")


def predict_sentiment(text):
    """
    Predict sentiment of a given text.
    
    Args:
        text (str): Input text/tweet
        
    Returns:
        dict: Contains sentiment, confidence, and probabilities
    """
    # Transform text using vectorizer
    vector = vectorizer.transform([text])
    
    # Get prediction
    prediction = model.predict(vector)[0]
    
    # Get probabilities
    probabilities = model.predict_proba(vector)[0]
    
    # Get confidence score
    confidence = probabilities[prediction] * 100
    
    # Map prediction to label
    sentiment_label = class_mapping[prediction]
    
    return {
        'text': text,
        'sentiment': sentiment_label,
        'confidence': round(confidence, 2),
        'probabilities': {
            'Negative': round(probabilities[0] * 100, 2),
            'Positive': round(probabilities[1] * 100, 2)
        }
    }


# Test the function
if __name__ == "__main__":
    # Test tweets
    test_tweets = [
        "i love this product its amazing",
        "this is terrible worst experience ever",
        "great service highly recommend",
        "im so disappointed wont buy again"
    ]
    
    print("\n" + "="*50)
    print("TESTING PREDICTION FUNCTION")
    print("="*50)
    
    for tweet in test_tweets:
        result = predict_sentiment(tweet)
        print(f"\nTweet: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Probabilities: {result['probabilities']}")
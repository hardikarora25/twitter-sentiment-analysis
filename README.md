# 🐦 Twitter Sentiment Analysis

A machine learning application that analyzes Twitter sentiment and classifies tweets as **Positive** or **Negative** using Natural Language Processing (NLP) and Logistic Regression.

![Twitter Sentiment Analysis](https://img.shields.io/badge/Twitter-Sentiment_Analysis-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-76.5%25-28A745?style=for-the-badge)

---

## 📋 Table of Contents

- [Features](#-features)
- [Live Demo](#-live-demo)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Screenshots](#-screenshots)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## ✨ Features

- **Single Tweet Analysis** - Analyze sentiment of individual tweets in real-time
- **Batch Analysis** - Upload CSV files for bulk sentiment analysis
- **Confidence Scores** - Get probability scores for each prediction
- **Interactive Dashboard** - User-friendly Streamlit web interface
- **Data Preprocessing** - Advanced text cleaning with NLTK (stopwords, lemmatization)
- **TF-IDF Vectorization** - Feature extraction with 5000 features
- **Export Results** - Download analysis results as CSV

---

## 🌐 Live Demo

Check out the live application: **[Twitter Sentiment Analysis App](https://your-app-url.streamlit.app)**

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| **Language** | Python 3.11 |
| **ML Library** | Scikit-learn |
| **NLP** | NLTK, TextBlob |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | Streamlit |
| **Model** | Logistic Regression + TF-IDF |

---

## 📁 Project Structure

twitter-sentiment-analysis/
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
├── .gitignore # Git ignore file
├── data/
│ ├── raw/ # Raw dataset
│ └── processed/ # Cleaned dataset
├── models/
│ ├── sentiment_model.pkl # Trained Logistic Regression model
│ ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
│ └── class_mapping.pkl # Class label mapping
├── notebooks/
│ ├── 01_data_cleaning.ipynb # Data preprocessing notebook
│ └── 02_model_training.ipynb # Model training notebook
└── src/
└── predict.py # Prediction script


---

## 📦 Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/hardikarora25/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis
2. Create Virtual Environment
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Download NLTK Data
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
🚀 Usage
Run the Streamlit App
streamlit run app.py
The app will open in your browser at http://localhost:8501

Run Prediction Script
python src/predict.py
Use in Your Code
from src.predict import predict_sentiment

result = predict_sentiment("I love this product!")
print(result)
# Output: {'text': 'I love this product!', 'sentiment': 'Positive', 'confidence': 85.2, ...}
📊 Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression + TF-IDF	76.5%	0.76	0.76	0.76
Naive Bayes + TF-IDF	75.1%	0.75	0.75	0.75
Training Data: 100,000 tweets (balanced classes)
Features: TF-IDF with 5000 max features

📸 Screenshots
Home Page
Home Page

Single Tweet Analysis
Single Analysis

Batch Analysis
Batch Analysis

Tip: Replace the placeholder URLs with actual screenshots of your app!

🔮 Future Improvements
[ ] Integrate Twitter API for live tweet analysis
[ ] Add BERT/RoBERTa for improved accuracy (~85-90%)
[ ] Implement negation handling for better context understanding
[ ] Add multi-language support
[ ] Include emoji sentiment analysis
[ ] Add historical trend analysis
[ ] Deploy as REST API using FastAPI
📄 License
This project is open source and available under the MIT License.

🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

📫 Connect with Me
GitHub: @hardikarora25
LinkedIn: Add your LinkedIn
Twitter: Add your Twitter
Email: add-your-email@example.com
<div align=“center”>

Made with ❤️ by Hardik Arora

⭐ Star this repo if you find it helpful!

</div>

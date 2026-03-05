import streamlit as st
import pandas as pd
import joblib
import os

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide"
)

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load('models/sentiment_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    class_mapping = joblib.load('models/class_mapping.pkl')
    return model, vectorizer, class_mapping

model, vectorizer, class_mapping = load_models()

# Title
st.title("🐦 Twitter Sentiment Analysis")
st.markdown("---")
# Sidebar
st.sidebar.header("📝 Menu")
option = st.sidebar.radio(
    "Choose Analysis Type:",
    ["Single Tweet", "Batch Analysis", "About"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info:**")
st.sidebar.info("- Logistic Regression + TF-IDF\n- Accuracy: ~76.5%")

# Single Tweet Analysis
if option == "Single Tweet":
    st.header("📌 Analyze Single Tweet")
    
    # Input box
    tweet_input = st.text_area(
        "Enter a tweet to analyze:",
        height=100,
        placeholder="Type your tweet here..."
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if tweet_input.strip():
            # Predict
            vector = vectorizer.transform([tweet_input])
            prediction = model.predict(vector)[0]
            probabilities = model.predict_proba(vector)[0]
            
            # Get sentiment label
            sentiment = class_mapping[prediction]
            confidence = probabilities[prediction] * 100
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Result")
                if sentiment == 'Positive':
                    st.success(f"**Sentiment:** {sentiment} ✅")
                else:
                    st.error(f"**Sentiment:** {sentiment} ❌")
                st.write(f"**Confidence:** {confidence:.2f}%")
            
            with col2:
                st.subheader("📈 Probabilities")
                st.metric("Negative", f"{probabilities[0]*100:.2f}%")
                st.metric("Positive", f"{probabilities[1]*100:.2f}%")
            
            # Progress bar
            st.subheader("Confidence Level")
            st.progress(confidence / 100)
        else:
            st.warning("Please enter a tweet!")

# Batch Analysis
elif option == "Batch Analysis":
    st.header("📊 Batch Analysis")
    st.write("Upload a CSV file with a 'text' column to analyze multiple tweets at once.")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Check if 'text' column exists
        if 'text' in df.columns:
            st.write(f"Loaded {len(df)} tweets")
            
            # Analyze button
            if st.button("Analyze All Tweets", type="primary"):
                with st.spinner("Analyzing... Please wait"):
                    # Predict for all rows
                    vectors = vectorizer.transform(df['text'].fillna(''))
                    predictions = model.predict(vectors)
                    probabilities = model.predict_proba(vectors)
                    
                    # Add results to dataframe
                    df['sentiment'] = [class_mapping[p] for p in predictions]
                    df['confidence'] = [probabilities[i][predictions[i]] * 100 for i in range(len(predictions))]
                    df['prob_negative'] = [probabilities[i][0] * 100 for i in range(len(predictions))]
                    df['prob_positive'] = [probabilities[i][1] * 100 for i in range(len(predictions))]
                    
                    st.success("✅ Analysis Complete!")
                    
                    # Show results
                    st.subheader("📋 Results Preview")
                    st.dataframe(df[['text', 'sentiment', 'confidence']].head(10))
                    
                    # Summary statistics
                    st.subheader("📈 Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Tweets", len(df))
                    with col2:
                        st.metric("Positive", len(df[df['sentiment'] == 'Positive']))
                    with col3:
                        st.metric("Negative", len(df[df['sentiment'] == 'Negative']))
                    
                    # Sentiment distribution chart
                    st.subheader("📊 Sentiment Distribution")
                    sentiment_counts = df['sentiment'].value_counts()
                    st.bar_chart(sentiment_counts)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv,
                        file_name="sentiment_results.csv",
                        mime="text/csv"
                    )
        else:
            st.error("CSV file must contain a 'text' column!")
# About Page
elif option == "About":
    st.header("📖 About This Project")
    
    st.write("""
    This is a **Twitter Sentiment Analysis** application built using Machine Learning.
    It can classify tweets as **Positive** or **Negative** with approximately 76% accuracy.
    """)
    
    st.subheader("🛠️ Technologies Used")
    st.markdown("""
    - **Programming Language:** Python 3.11
    - **Machine Learning:** Scikit-learn (Logistic Regression + TF-IDF)
    - **Frontend:** Streamlit
    - **Data Processing:** Pandas, NumPy, NLTK
    """)
    
    st.subheader("📊 Model Details")
    st.info("""
    - **Algorithm:** Logistic Regression
    - **Feature Extraction:** TF-IDF (5000 features)
    - **Training Data:** 100,000 tweets (balanced)
    - **Accuracy:** ~76.5%
    """)
    
    st.subheader("📁 Project Structure")
    st.code("""
    sentiment/
    ├── app.py                 # Streamlit application
    ├── models/                # Trained models
    ├── src/                   # Python scripts
    ├── notebooks/             # Jupyter notebooks
    ├── data/                  # Dataset files
    └── requirements.txt       # Dependencies
    """, language="plaintext")
    
    st.subheader("🚀 How to Run")
    st.code("streamlit run app.py", language="bash")
    
    st.markdown("---")
    st.write("**Made with ❤️ for sentiment analysis**")
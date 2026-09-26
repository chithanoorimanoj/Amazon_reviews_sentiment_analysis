import streamlit as st
import joblib
import pickle
import re
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="centered"
)


# --------------------------------------------------
# NLTK SETUP
# --------------------------------------------------

@st.cache_resource
def setup_nltk():

    packages = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab")
    ]

    for path, package in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


setup_nltk()


# --------------------------------------------------
# TEXT CLEANING
# Same basic preprocessing used in the notebook
# --------------------------------------------------

def clean_text(doc):

    doc = str(doc)

    # Remove characters other than letters and periods
    regex = r"[^a-zA-Z.]"
    doc = re.sub(regex, " ", doc)

    # Convert to lowercase
    doc = doc.lower()

    # Tokenization
    tokens = nltk.word_tokenize(doc)

    # Stopword removal
    stop_words = set(stopwords.words("english"))

    filtered_tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()

    lemmatized_tokens = [
        lemmatizer.lemmatize(token)
        for token in filtered_tokens
    ]

    return " ".join(lemmatized_tokens)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():

    # First try model.joblib
    if os.path.exists("model.joblib"):

        try:

            artifact = joblib.load("model.joblib")

            # Your notebook saved a dictionary containing "model"
            if isinstance(artifact, dict) and "model" in artifact:
                return artifact["model"]

            # If joblib directly contains the model
            return artifact

        except Exception as e:
            st.error(f"Error loading model.joblib: {e}")
            st.stop()

    # If model.joblib doesn't exist, try model.pkl
    elif os.path.exists("model.pkl"):

        try:

            with open("model.pkl", "rb") as file:
                model = pickle.load(file)

            return model

        except Exception as e:
            st.error(f"Error loading model.pkl: {e}")
            st.stop()

    else:

        st.error(
            "Model file not found. "
            "Please upload model.joblib or model.pkl."
        )

        st.stop()


model = load_model()


# --------------------------------------------------
# SENTIMENT MAPPING
# --------------------------------------------------

sentiment_map = {
    1: "Negative",
    2: "Negative",
    3: "Neutral",
    4: "Positive",
    5: "Positive"
}


# --------------------------------------------------
# APPLICATION UI
# --------------------------------------------------

st.title("🛒 Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon product review and the machine learning "
    "model will predict its sentiment."
)

st.divider()


review = st.text_area(
    "Enter your Amazon Review",
    placeholder="Example: The product quality is excellent and I really liked it.",
    height=180
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔍 Analyze Review", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a review first.")

    else:

        with st.spinner("Analyzing review..."):

            cleaned_review = clean_text(review)

            prediction = model.predict([cleaned_review])[0]

            try:
                prediction = int(prediction)
            except:
                pass

            sentiment = sentiment_map.get(
                prediction,
                str(prediction)
            )

        st.divider()

        st.subheader("Prediction Result")

        if sentiment == "Positive":

            st.success("😊 Positive Review")

        elif sentiment == "Negative":

            st.error("😞 Negative Review")

        else:

            st.warning("😐 Neutral Review")


        st.write(f"**Predicted Score:** {prediction}")
        st.write(f"**Sentiment:** {sentiment}")

        # Probability if model supports it
        try:

            probabilities = model.predict_proba(
                [cleaned_review]
            )[0]

            confidence = max(probabilities) * 100

            st.metric(
                "Prediction Confidence",
                f"{confidence:.2f}%"
            )

        except:

            pass


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("About the Project")

    st.write(
        """
        **Amazon Review Sentiment Analysis**

        Machine Learning model used:

        • TF-IDF Vectorization  
        • Logistic Regression  
        • Text preprocessing  
        • Sentiment classification
        """
    )

    st.info(
        "The model predicts Amazon review ratings "
        "from 1 to 5 and maps them to Negative, "
        "Neutral and Positive sentiment."
    )
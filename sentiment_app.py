import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

classifier = load_model()

# Title
st.title("🎬 Movie Review Sentiment Analyzer")
st.write("Enter a movie review below to analyze its sentiment!")

# Input
user_input = st.text_area("Your review:", height=150)

# Button
if st.button("Analyze Sentiment", type="primary"):
    if user_input:
        result = classifier(user_input)[0]

        if result["label"] == "POSITIVE":
            st.success(f"Sentiment: {result['label']} 😊")
        else:
            st.error(f"Sentiment: {result['label']} 😞")

        st.metric("Confidence", f"{result['score']:.2%}")
        st.progress(result["score"])
    else:
        st.warning("Please enter text first!")

# Sidebar examples
st.sidebar.header("Examples")

examples = [
    "This movie was amazing!",
    "Worst film ever.",
    "It was okay."
]

for ex in examples:
    st.sidebar.write(ex)

# Footer
st.divider()
st.caption("Built with Streamlit + Hugging Face Transformers")
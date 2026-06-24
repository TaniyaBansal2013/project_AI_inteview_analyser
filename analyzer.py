import streamlit as st
import speech_recognition as sr
import whisper
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="AI Interview Analyzer")

st.title("AI Interview Analyzer")
st.write("Click the button below to begin your interview.")

if st.button("Start Interview"):
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    st.write("INTRODUCE YOURSELF")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        st.info("Recording... Start Speaking (30 seconds)")
        audio = recognizer.record(source, duration=30)

    with open("introduction.wav", "wb") as file:
        file.write(audio.get_wav_data())

    st.success("Recording Completed!")

    st.info("Analyzing your response... Please wait.")
    model = whisper.load_model("base")
    result = model.transcribe("introduction.wav", language="en")
    text = result["text"]

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    def preprocess(text):
        final_text = text.lower()
        words = word_tokenize(final_text)

        words = [word for word in words if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]

        return " ".join(words)
    cleaned_text = preprocess(text)
    analyser = SentimentIntensityAnalyzer()
    scores = analyser.polarity_scores(text)

    total_words = len(cleaned_text.split())
    unique_words = len(set(cleaned_text.split()))
    if total_words > 0:
        vocab_score = (unique_words / total_words) * 100
    else:
        vocab_score = 0

    filler_words = [
        "um", "uhm", "actually", "basically",
        "uh", "like", "you know", "hmmm"
    ]

    spoken_words = word_tokenize(text.lower())

    filler_count = 0
    found_fillers = []
    for word in spoken_words:
        if word in filler_words:
            filler_count += 1
            found_fillers.append(word)

    keywords = [
        "python", "ml", "ai",
        "artificial intelligence",
        "github",
        "machine learning",
        "numpy",
        "pandas",
        "libraries",
        "project",
        "sql",
        "java"
    ]

    keywords_count = 0
    found_keywords = []

    for keyword in keywords:
        if keyword.lower() in text.lower():
            keywords_count += 1
            found_keywords.append(keyword)

    if scores["compound"] >= 0.5:
        sentiment_score = 20
    elif scores["compound"] >= 0:
        sentiment_score = 15
    else:
        sentiment_score = 5

    if round(vocab_score, 2) >= 90:
        vocabulary_score = 20
    elif round(vocab_score, 2) >= 75:
        vocabulary_score = 15
    else:
        vocabulary_score = 10

    if total_words >= 50:
        words_score = 20
    elif total_words >= 35:
        words_score = 10
    else:
        words_score = 5

    if keywords_count >= 5:
        keywords_score = 20
    elif keywords_count >= 3:
        keywords_score = 15
    elif keywords_count >= 1:
        keywords_score = 10
    else:
        keywords_score = 0

    if filler_count == 0:
        filler_score = 20
    elif filler_count <= 2:
        filler_score = 10
    else:
        filler_score = 5

    final_score = (
        sentiment_score +
        vocabulary_score +
        words_score +
        keywords_score +
        filler_score
    )

    st.header("AI Interview Report")

    st.subheader("Transcript")
    st.write(text)

    st.write("Positive :", scores["pos"])
    st.write("Neutral :", scores["neu"])
    st.write("Negative :", scores["neg"])
    st.write("Overall :", scores["compound"])

    st.write("Total Words :", total_words)
    st.write("Unique Words :", unique_words)
    st.write("Vocabulary Richness :", round(vocab_score, 2), "%")

    st.subheader("Filler Words")

    if filler_count > 0:
        st.write(found_fillers)
    else:
        st.write("No filler words found.")

    st.subheader("Technical Keywords")

    if keywords_count > 0:
        st.write(found_keywords)
    else:
        st.write("No technical keywords found.")

    st.subheader("Scores")

    st.write("Sentiment Score :", sentiment_score, "/20")
    st.write("Vocabulary Score :", vocabulary_score, "/20")
    st.write("Word Count Score :", words_score, "/20")
    st.write("Technical Score :", keywords_score, "/20")
    st.write("Fluency Score :", filler_score, "/20")

    st.metric("Interview Score", f"{final_score}/100")

    if final_score >= 85:
        st.success("SELECTED!")
    elif final_score >= 70:
        st.warning("SELECTED FOR NEXT ROUND!")
    else:
        st.error("BETTER LUCK NEXT TIME!")

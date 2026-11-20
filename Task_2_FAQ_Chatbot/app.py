import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import FAQS

st.set_page_config(page_title="AI Internship FAQ Chatbot", page_icon="🤖")
st.title("🤖 AI Internship FAQ Chatbot")
st.write("Ask a question about the AI internship tasks and submission requirements.")

questions = [item["question"] for item in FAQS]
vectorizer = TfidfVectorizer(stop_words="english")
faq_matrix = vectorizer.fit_transform(questions)

if "chat" not in st.session_state:
    st.session_state.chat = []

with st.form("question_form", clear_on_submit=True):
    user_question = st.text_input("Your question", placeholder="e.g. How many tasks do I need to complete?")
    submitted = st.form_submit_button("Ask")

if submitted:
    if not user_question.strip():
        st.warning("Please type a question.")
    else:
        user_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(user_vector, faq_matrix)[0]
        best_index = scores.argmax()
        best_score = scores[best_index]

        if best_score < 0.12:
            answer = "Sorry, I could not find a close answer in my FAQ database. Try asking about the four tasks or submission requirements."
        else:
            answer = FAQS[best_index]["answer"]

        st.session_state.chat.append(("You", user_question))
        st.session_state.chat.append(("Bot", answer))

for speaker, message in st.session_state.chat:
    with st.chat_message("user" if speaker == "You" else "assistant"):
        st.write(message)

if st.button("Clear chat"):
    st.session_state.chat = []
    st.rerun()

with st.expander("Questions you can try"):
    for question in questions:
        st.write("• " + question)

st.caption("NLP method: TF-IDF vectorization + cosine similarity.")

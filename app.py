import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import random
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.set_page_config(page_title="MCQ Generator", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main-container {
            background-color: #1e1e1e;
            color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
        }
        .file-uploader {
            border: 2px dashed #4CAF50;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            color: #4CAF50;
        }
        .generate-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .generate-button:hover {
            background-color: #45a049;
        }
        .next-button {
            float: right;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .next-button:hover {
            background-color: #45a049;
        }
        .centered-text {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        .correct-answer {
            color: green;
            font-weight: bold;
        }
        .wrong-answer {
            color: red;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# App Header
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.title("📚 MCQ Generator")
st.write("Upload a PDF to generate an interactive MCQ based on its content.")

# 1) File Upload
uploaded_file = st.file_uploader(
    '<div class="file-uploader">Drop your PDF here or click to browse.</div>',
    type=["pdf"],
    label_visibility="collapsed",
)

temp_file_path = "temp_uploaded_file.pdf"

if uploaded_file:
    if "pdf_processed" not in st.session_state:
        with st.spinner("Uploading and processing your PDF..."):
            try:
                # Save the uploaded file
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success("File uploaded and saved successfully!")
            except Exception as e:
                st.error(f"Error saving file: {e}")
                st.stop()

        # Analyze the PDF content (only once)
        with st.spinner("Analyzing the PDF content..."):
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            embeddings = OpenAIEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)

            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=retriever
            )

            # Store the QA chain in session state
            st.session_state["qa_chain"] = qa_chain
            st.session_state["pdf_processed"] = True
            st.success("PDF processed successfully! Click 'Generate MCQ' to start.")

# Helper function to create MCQ answers
def generate_mcq(question, correct_answer, qa_chain):
    options = [correct_answer]
    while len(options) < 4:
        fake_answer = qa_chain.run(
            f"Provide a plausible but incorrect answer for this question: {question}. "
            "Avoid repeating previous options."
        )
        if fake_answer not in options and len(fake_answer.strip()) > 0:
            options.append(fake_answer)
    random.shuffle(options)
    return options

# Generate quiz button
if st.session_state.get("pdf_processed") and st.button("Generate MCQ", key="generate", help="Start generating the mcq"):
    if "questions" not in st.session_state:
        with st.spinner("Generating your MCQ... Please wait."):
            st.session_state["questions"] = []
            st.session_state["score"] = 0
            st.session_state["current_question"] = 0

            # Generate exactly 4 MCQs
            for _ in range(4):
                question = st.session_state["qa_chain"].run(
                    "Create a multiple-choice question based on the document content. Provide only the question."
                )
                correct_answer = st.session_state["qa_chain"].run(
                    f"Provide the correct answer for this question: {question}. "
                    "Provide only the correct answer."
                )
                options = generate_mcq(question, correct_answer, st.session_state["qa_chain"])

                st.session_state["questions"].append((question, options, correct_answer))

# Show MCQs to user
if "questions" in st.session_state:
    current_q = st.session_state["current_question"]
    if current_q < len(st.session_state["questions"]):
        question, options, correct_answer = st.session_state["questions"][current_q]
        st.write(f"**Question {current_q + 1}:** {question}")

        selected_option = st.radio("Choose an option:", options, key=f"q{current_q}")
        
        if st.button("Submit Answer", key=f"submit{current_q}"):
            if selected_option == correct_answer:
                st.session_state["score"] += 1
                st.markdown("<span class='correct-answer'>Correct!</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span class='wrong-answer'>Wrong!</span>", unsafe_allow_html=True)

        # Add a Next button to move to the next question
        if st.button("Next", key=f"next{current_q}"):
            st.session_state["current_question"] += 1
    else:
        # All questions have been answered
        total_q = len(st.session_state["questions"])
        score_percentage = (st.session_state["score"] / total_q) * 100
        st.markdown("<div class='centered-text'>Quiz Completed!</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='centered-text' style='font-weight:bold;'>Your Score: {score_percentage:.2f}%</div>",
            unsafe_allow_html=True,
        )

        # Display questions and correct answers
        st.markdown("<div class='centered-text'>Review:</div>", unsafe_allow_html=True)
        for i, (question, _, correct_answer) in enumerate(st.session_state["questions"]):
            st.write(f"**Question {i + 1}:** {question}")
            st.write(f"**Correct Answer:** {correct_answer}")

# Clean up temporary file
if os.path.isfile(temp_file_path):
    os.remove(temp_file_path)
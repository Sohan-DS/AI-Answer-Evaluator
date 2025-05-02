import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import pdfplumber
import docx

# Load SentenceTransformer for similarity evaluation
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load a summarizer model (open-source)
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Dummy question and expected answer (can be dynamic or from DB)
question = "Explain the process of photosynthesis."
expected_answer = """
Photosynthesis is the process by which green plants use sunlight to synthesize nutrients from carbon dioxide and water. 
It involves the green pigment chlorophyll and generates oxygen as a byproduct. The overall chemical equation is 
6CO2 + 6H2O + sunlight → C6H12O6 + 6O2.
"""

# Function to extract text from uploaded files
def extract_text(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with pdfplumber.open(uploaded_file) as pdf:
            return " ".join(page.extract_text() for page in pdf.pages if page.extract_text())
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    else:
        return None

# Function to evaluate the answer
def evaluate_answer(student_answer, reference_answer):
    # Similarity score (0-100)
    emb1 = model.encode(student_answer, convert_to_tensor=True)
    emb2 = model.encode(reference_answer, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(emb1, emb2).item() * 100

    # Score interpretation
    if similarity_score > 85:
        remark = "Excellent answer"
        marks = 10
    elif similarity_score > 70:
        remark = "Good answer"
        marks = 8
    elif similarity_score > 50:
        remark = "Average answer"
        marks = 5
    else:
        remark = "Poor or irrelevant answer"
        marks = 2

    return marks, round(similarity_score, 2), remark

# --- Streamlit UI ---
st.title("📘 Long Answer Evaluator (Free, Open-Source)")

st.markdown("**Question:**")
st.info(question)

option = st.radio("How do you want to submit your answer?", ["Type here", "Upload document"])

student_answer = ""

if option == "Type here":
    student_answer = st.text_area("Type your answer here:")
elif option == "Upload document":
    uploaded_file = st.file_uploader("Upload your answer (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
    if uploaded_file:
        student_answer = extract_text(uploaded_file)
        st.success("File uploaded and text extracted!")

if st.button("Evaluate Answer") and student_answer.strip():
    with st.spinner("Evaluating your answer..."):
        summary = summarizer(student_answer[:2000])[0]['summary_text']
        marks, score, remark = evaluate_answer(summary, expected_answer)
        st.markdown(f"**Marks:** {marks}/10")
        st.markdown(f"**Similarity Score:** {score}%")
        st.markdown(f"**Feedback:** {remark}")
else:
    st.warning("Please provide an answer to evaluate.")
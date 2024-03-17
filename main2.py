import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the summarization model
checkpoint ="facebook/bart-large-cnn"
model = pipeline('summarization', model=checkpoint)

# Streamlit UI
st.title("Text Summarizer using LLM")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text


# Function to summarize text
def summarize_text(text):
    summary = model(text, min_length=256, max_length=512, do_sample=True)[0]['summary_text']
    return summary

# Radio button for selecting input format
input_format = st.selectbox("Select input format:", ('Text', 'PDF'))


# PDF input box for the document to be summarized
if input_format == 'Text':
    uploaded_file = st.file_uploader("Upload a text document (.txt)", type="txt")
    if uploaded_file is not None:
        text = uploaded_file.read().decode("utf-8")
        st.subheader("Original Text:")
        st.write(text)

        if st.button("Summarize"):
            # Generate the summary
            summary = summarize_text(text)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)

elif input_format == 'PDF':
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        st.subheader("Original Text:")
        st.write(text)

        if st.button("Summarize"):
            # Generate the summary
            summary = summarize_text(text)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)

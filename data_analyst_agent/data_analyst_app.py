import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
from together import Together
import io
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    logger.error("TOGETHER_API_KEY not set")
    st.error("API key missing. Please contact the administrator.")
    st.stop()

# Initialize Together.ai client
client = Together(api_key=TOGETHER_API_KEY)

# Streamlit UI
st.title("Data Analyst App")
st.write("Upload a file and ask questions about its data.")
uploaded_file = st.file_uploader("Upload file", type=["docx", "txt", "xlsx", "csv", "pdf", "png"])
question = st.text_input("Ask a question about the data")
if st.button("Submit"):
    if not uploaded_file or not question:
        st.warning("Please upload a file and enter a question.")
    else:
        try:
            logger.info(f"Processing file: {uploaded_file.name}")
            # Process Excel/CSV
            if uploaded_file.type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "text/csv"]:
                df = pd.read_excel(uploaded_file) if uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" else pd.read_csv(uploaded_file)
                st.write("Data Preview:", df.head())
                st.write("Summary:", df.describe())
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[{"role": "user", "content": f"Data: {df.to_string()}\nQuestion: {question}"}]
                )
                st.success("Answer:")
                st.write(response.choices[0].message.content)
                # Optional visualization
                if st.checkbox("Generate Plot"):
                    column = st.selectbox("Select column to plot", df.columns)
                    plt.figure(figsize=(10, 6))
                    df[column].plot(kind="bar")
                    plt.title(f"{column} Trend")
                    plt.xlabel("Index")
                    plt.ylabel(column)
                    st.pyplot(plt)
            # Process PNG
            elif uploaded_file.type == "image/png":
                image = Image.open(uploaded_file)
                text = pytesseract.image_to_string(image)
                st.write("Extracted Text:", text)
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[{"role": "user", "content": f"Extracted text: {text}\nQuestion: {question}"}]
                )
                st.success("Answer:")
                st.write(response.choices[0].message.content)
            # Process PDF
            elif uploaded_file.type == "application/pdf":
                pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                text = ""
                for page in pdf:
                    text += page.get_text()
                st.write("Extracted Text:", text[:500] + "..." if len(text) > 500 else text)
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[{"role": "user", "content": f"Extracted text: {text}\nQuestion: {question}"}]
                )
                st.success("Answer:")
                st.write(response.choices[0].message.content)
            # Process Text/Docx
            elif uploaded_file.type in ["text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                text = uploaded_file.read().decode("utf-8") if uploaded_file.type == "text/plain" else "Docx processing requires python-docx"
                st.write("Extracted Text:", text[:500] + "..." if len(text) > 500 else text)
                response = client.chat.completions.create(
                    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    messages=[{"role": "user", "content": f"Extracted text: {text}\nQuestion: {question}"}]
                )
                st.success("Answer:")
                st.write(response.choices[0].message.content)
            else:
                st.error("Unsupported file type.")
                logger.warning(f"Unsupported file type: {uploaded_file.type}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
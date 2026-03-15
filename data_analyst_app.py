import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
import logging
from PIL import Image
import pytesseract
import fitz
from docx import Document
import openpyxl
import requests
import time
from groq import Groq

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Universal Data Analyst App", layout="wide")

# Setup API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set. Please set it in Render's Environment Variables.")
    st.stop()

# --- HELPER FUNCTIONS ---

def custom_describe(df):
    """Generates an expanded statistical summary for any DataFrame."""
    desc = df.describe(include='all')
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        desc.loc['sum'] = df[numerical_cols].sum()
    return desc

def detect_columns(df):
    """
    Dynamically maps columns based on domain-agnostic keywords and data types.
    This allows the app to handle Flights, Sales, Medical, or any other data.
    """
    columns_lower = [col.strip().lower() for col in df.columns]
    columns = {
        'category': None,
        'sub_category': None,
        'profit': None,
        'sales': None,
        'region': None,
        'date': None,
        'customer': None,
        'id': None
    }
    
    for idx, col in enumerate(columns_lower):
        # 1. Category Mapping
        if 'category' in col and 'sub' not in col:
            columns['category'] = df.columns[idx]
        elif 'sub' in col and 'category' in col or any(x in col for x in ['item', 'type', 'species', 'model']):
            columns['sub_category'] = df.columns[idx]
        
        # 2. Performance/Metric 1 (Formerly 'Sales')
        elif any(x in col for x in ['sales', 'revenue', 'amount', 'distance', 'magnitude', 'total']):
            if 'id' not in col:
                columns['sales'] = df.columns[idx]
        
        # 3. Performance/Metric 2 (Formerly 'Profit')
        elif any(x in col for x in ['profit', 'margin', 'delay', 'score', 'rate', 'value']):
            columns['profit'] = df.columns[idx]
            
        # 4. Location Mapping
        elif any(x in col for x in ['region', 'area', 'city', 'origin', 'dest', 'state']):
            columns['region'] = df.columns[idx]
            
        # 5. Time Mapping
        elif any(x in col for x in ['date', 'time', 'year', 'month', 'timestamp']):
            columns['date'] = df.columns[idx]
            
        # 6. Entity Mapping (Formerly 'Customer')
        elif any(x in col for x in ['customer', 'client', 'carrier', 'name', 'subject', 'patient']):
            columns['customer'] = df.columns[idx]
            
        elif 'id' in col:
            columns['id'] = df.columns[idx]
            
    # Universal Fallback: If keywords fail, pick the best numeric columns
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not columns['sales'] and len(num_cols) > 0:
        for c in num_cols:
            if 'id' not in c.lower():
                columns['sales'] = c
                break
    if not columns['profit'] and len(num_cols) > 1:
        for c in num_cols:
            if c != columns['sales'] and 'id' not in c.lower():
                columns['profit'] = c
                break
                
    return columns

def load_file(uploaded_file):
    """Supports CSV, Excel, PDF, DOCX, TXT, and Images (OCR)."""
    try:
        file_type = uploaded_file.type
        logger.info(f"Detected file type: {file_type}")
        if file_type == "text/plain":
            return uploaded_file.read().decode('utf-8', errors='ignore')
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            file_bytes = uploaded_file.read()
            xl = pd.ExcelFile(io.BytesIO(file_bytes), engine='openpyxl')
            data = {sn: pd.read_excel(io.BytesIO(file_bytes), sheet_name=sn) for sn in xl.sheet_names}
            return data
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            return {'data': df}
        elif file_type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            return "\n".join([page.get_text() for page in doc])
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            return "\n".join([p.text for p in doc.paragraphs])
        elif file_type == "image/png" or file_type == "image/jpeg":
            return pytesseract.image_to_string(Image.open(uploaded_file))
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

# --- CORE ANALYSIS ENGINE ---

def analyze_dataframe(data_dict):
    try:
        analysis_results = {}
        figures = []
        
        # Preserve specific E-commerce Merging Logic for Superstore datasets
        if 'ListOfOrders' in data_dict and 'OrderBreakdown' in data_dict:
            try:
                lo, ob = data_dict['ListOfOrders'].copy(), data_dict['OrderBreakdown'].copy()
                lo.columns = [c.strip().lower() for c in lo.columns]
                ob.columns = [c.strip().lower() for c in ob.columns]
                if 'order id' in lo.columns and 'order id' in ob.columns:
                    lo['order id'] = lo['order id'].astype(str).str.strip()
                    ob['order id'] = ob['order id'].astype(str).str.strip()
                    ob_agg = ob.groupby('order id').agg({'sales': 'sum', 'profit': 'sum'}).reset_index()
                    data_dict['Merged'] = pd.merge(lo, ob_agg, on='order id', how='outer').fillna(0)
            except Exception as e:
                logger.error(f"Merge error: {e}")

        for sheet_name, df in data_dict.items():
            plt.clf() # Ensure fresh canvas for every chart
            df = df.copy()
            columns = detect_columns(df)
            
            # Numeric conversion to avoid plotting errors
            for k in ['sales', 'profit']:
                if columns[k]: 
                    df[columns[k]] = pd.to_numeric(df[columns[k]], errors='coerce').fillna(0)

            # Context generation for AI (Full columns + sample rows)
            data_sample = df.head(3).to_string(index=False)
            all_cols = ", ".join(df.columns.tolist())

            # --- Visualizations ---
            num_cols = df.select_dtypes(include=['number']).columns
            cat_cols = df.select_dtypes(include=['object', 'category']).columns

            # 1. First Numeric Distribution
            if not num_cols.empty:
                c = num_cols[0]
                fig, ax = plt.subplots(figsize=(8, 5))
                # Sample to 10k rows to ensure performance on large datasets
                sns.histplot(df[c].dropna().head(10000), kde=True, ax=ax, color='teal')
                plt.title(f'Distribution: {c}')
                figures.append((fig, f"Numerical analysis of {c}"))

            # 2. First Categorical Frequency
            if not cat_cols.empty:
                c = cat_cols[0]
                fig, ax = plt.subplots(figsize=(8, 5))
                # CRITICAL: Always use nlargest(10) to prevent empty/cluttered charts
                top_10 = df[c].value_counts().nlargest(10)
                sns.barplot(x=top_10.values, y=top_10.index, ax=ax, palette='viridis')
                plt.title(f'Top 10: {c}')
                figures.append((fig, f"Frequency analysis of {c}"))

            # Summaries (Safely initialized to avoid UnboundLocalError)
            total_v1 = df[columns['sales']].sum() if columns['sales'] else 0
            total_v2 = df[columns['profit']].sum() if columns['profit'] else 0
            unique_ent = df[columns['customer']].nunique() if columns['customer'] else 0
            
            # Temporal Trend Logic
            trend_str = "No temporal trend detected."
            if columns['date'] and columns['sales']:
                try:
                    df[columns['date']] = pd.to_datetime(df[columns['date']], errors='coerce')
                    df_t = df.dropna(subset=[columns['date']])
                    if not df_t.empty:
                        df_t['month_year'] = df_t[columns['date']].dt.to_period('M')
                        trend_agg = df_t.groupby('month_year')[columns['sales']].sum().nlargest(5)
                        trend_str = trend_agg.to_string()
                except: pass

            analysis_results[sheet_name] = {
                'column_list': all_cols,
                'sample_data': data_sample,
                'metric_1': {'name': columns['sales'], 'total': total_v1},
                'metric_2': {'name': columns['profit'], 'total': total_v2},
                'entities': {'name': columns['customer'], 'count': unique_ent},
                'trend': trend_str,
                'columns': df.columns.tolist(),
                'mapping': columns
            }
            plt.close('all') # Cleanup
            
        return analysis_results, figures
    except Exception as e:
        logger.error(f"Global analysis error: {e}")
        return None, []

# --- AI ANALYST CORE ---

def answer_question(context, question, retries=3, delay=10):
    """Uses Groq to act as a smart, domain-aware Data Analyst."""
    client = Groq(api_key=GROQ_API_KEY)
    
    prompt = f"""You are a professional Data Analyst. 
First, identify the SUBJECT of the data (e.g., flight delays, medical stats, or business sales) 
using the provided Sample Rows and Column Names. 
Then, answer the user's question accurately using ONLY the context provided.

DATASET CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
1. Briefly acknowledge the domain (e.g., "Based on the flight records provided...")
2. Use the actual column names from the data.
3. If asked for a total or average, refer to the metric summaries provided.
4. Format numbers with commas (e.g., 1,234,567.89).
5. If the specific answer is not in the context, say: "The provided summary doesn't contain that specific level of detail."
"""

    for attempt in range(retries):
        try:
            res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024
            )
            return res.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return "The AI analyst is currently overloaded. Please refer to the raw data and charts above."

# --- STREAMLIT UI ---

st.title("📊 Universal Data Analyst")
st.markdown("Upload **any** structured or unstructured file for intelligent analysis and chat.")

uploaded_file = st.file_uploader("Upload File (CSV, Excel, PDF, Docx, Image)", type=["csv", "xlsx", "pdf", "docx", "txt", "png", "jpg"])

if uploaded_file:
    st.success(f"File `{uploaded_file.name}` uploaded!")
    try:
        content = load_file(uploaded_file)
        
        # HANDLE TABULAR DATA
        if isinstance(content, dict):
            results, figures = analyze_dataframe(content)
            if results:
                global_ai_context = ""
                for sheet_name, data in results.items():
                    # Build rich context for the AI
                    global_ai_context += (
                        f"Source: {sheet_name}\n"
                        f"All Columns: {data['column_list']}\n"
                        f"Sample Rows:\n{data['sample_data']}\n"
                        f"Metric 1 ({data['metric_1']['name']}): Total {data['metric_1']['total']}\n"
                        f"Metric 2 ({data['metric_2']['name']}): Total {data['metric_2']['total']}\n"
                        f"Entities ({data['entities']['name']}): {data['entities']['count']} unique entries\n"
                        f"Trends: {data['trend']}\n---\n"
                    )
                    st.subheader(f"Data Preview: {sheet_name}")
                    st.dataframe(content[sheet_name].head(10))
                
                st.write("### 📈 Visual Insights")
                if figures:
                    ui_cols = st.columns(2)
                    for idx, (fig, cap) in enumerate(figures):
                        with ui_cols[idx % 2]:
                            st.pyplot(fig)
                            st.caption(cap)
                
                st.write("### 💬 Ask the Analyst")
                user_q = st.text_input("Ask anything (e.g., 'Identify the subject and summarize', 'What is the highest value in [column]?')")
                if st.button("Submit Question") and user_q:
                    with st.spinner("AI is analyzing domain and data..."):
                        st.markdown("#### Analyst Response")
                        st.info(answer_question(global_ai_context, user_q))
            else:
                st.error("Analysis failed. Check your file for empty or corrupted data.")
                
        # HANDLE UNSTRUCTURED DATA
        elif isinstance(content, str):
            st.write("### 📄 Extracted Content Preview")
            st.text_area("Content", content[:2000], height=300)
            
            if "image" in uploaded_file.type:
                st.image(uploaded_file, caption="Uploaded Document", use_container_width=True)
                
            st.write("### 💬 Ask about this Document")
            user_q = st.text_input("Enter your question:")
            if st.button("Query Text") and user_q:
                with st.spinner("AI is reading document..."):
                    # Pass the first 4000 chars as context for AI
                    st.markdown(answer_question(content[:4000], user_q))
                
    except Exception as e:
        st.error(f"Processing Error: {str(e)}")
else:
    st.info("Please upload a file to begin.")
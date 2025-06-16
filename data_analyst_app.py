import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
from together import Together
import logging
from PIL import Image
import pytesseract
import fitz
from docx import Document
import openpyxl
import requests
import time

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit app configuration
st.set_page_config(page_title="Data Analyst App", layout="wide")

# Setup API Key
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    st.error("TOGETHER_API_KEY not set. Please set it in Render's Environment Variables.")
    st.stop()
client = Together(api_key=TOGETHER_API_KEY)

# Custom Describe Function
def custom_describe(df):
    desc = df.describe(include='all')
    numerical_cols = df.select_dtypes(include=['number']).columns
    if len(numerical_cols) > 0:
        desc.loc['sum'] = df[numerical_cols].sum()
    return desc

# Detect Column Types Dynamically
def detect_columns(df):
    df.columns = [col.strip().lower() for col in df.columns]
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
    for col in df.columns:
        col_lower = col.lower()
        if 'category' in col_lower and 'sub' not in col_lower:
            columns['category'] = col
        elif 'sub' in col_lower and 'category' in col_lower:
            columns['sub_category'] = col
        elif 'profit' in col_lower or 'margin' in col_lower:
            columns['profit'] = col
        elif 'sales' in col_lower or 'revenue' in col_lower:
            columns['sales'] = col
        elif 'region' in col_lower or 'area' in col_lower:
            columns['region'] = col
        elif 'date' in col_lower or 'time' in col_lower:
            columns['date'] = col
        elif 'customer' in col_lower or 'client' in col_lower:
            columns['customer'] = col
        elif 'id' in col_lower:
            columns['id'] = col
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if not columns['sales'] and numerical_cols.size > 0:
        for col in numerical_cols:
            if df[col].max() > 0:
                columns['sales'] = col
                break
    if not columns['profit'] and numerical_cols.size > 0:
        for col in numerical_cols:
            if col != columns['sales']:
                columns['profit'] = col
                break
    logger.info(f"Detected columns: {columns}")
    return columns

# Load File (Modified for Streamlit without File I/O)
def load_file(uploaded_file):
    try:
        file_type = uploaded_file.type
        logger.info(f"Detected file type: {file_type}")
        if file_type == "text/plain":
            return uploaded_file.read().decode('utf-8', errors='ignore')
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            xl = pd.ExcelFile(uploaded_file, engine='openpyxl')
            data = {}
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(uploaded_file, sheet_name=sheet_name, engine='openpyxl')
                logger.info(f"Columns in sheet '{sheet_name}': {df.columns.tolist()}")
                data[sheet_name] = df
            return data
        elif file_type == "text/csv":
            df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')
            logger.info(f"Columns in CSV: {df.columns.tolist()}")
            return {'data': df}
        elif file_type == "application/pdf":
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "\n".join([page.get_text() for page in doc])
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = "\n".join([p.text for p in doc.paragraphs])
            logger.info(f"Extracted {len(text)} characters from DOCX")
            return text
        elif file_type == "image/png":
            text = pytesseract.image_to_string(Image.open(uploaded_file))
            logger.info(f"Extracted {len(text)} characters from image via OCR")
            return text
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        raise

# DataFrame Analysis (Memory-Optimized)
def analyze_dataframe(data_dict):
    try:
        analysis_results = {}
        figures = []
        merged_df = None
        if 'ListOfOrders' in data_dict and 'OrderBreakdown' in data_dict:
            list_orders = data_dict['ListOfOrders'].copy()
            order_breakdown = data_dict['OrderBreakdown'].copy()
            list_orders.columns = [col.strip().lower() for col in list_orders.columns]
            order_breakdown.columns = [col.strip().lower() for col in order_breakdown.columns]
            if 'order id' in list_orders.columns and 'order id' in order_breakdown.columns:
                try:
                    list_orders['order id'] = list_orders['order id'].astype(str).str.strip().str.replace(r'^\D+', '', regex=True)
                    order_breakdown['order id'] = order_breakdown['order id'].astype(str).str.strip().str.replace(r'^\D+', '', regex=True)
                    order_breakdown['sales'] = pd.to_numeric(order_breakdown['sales'], errors='coerce').fillna(0)
                    order_breakdown['profit'] = pd.to_numeric(order_breakdown['profit'], errors='coerce').fillna(0)
                    order_breakdown_agg = order_breakdown.groupby('order id').agg({
                        'sales': 'sum',
                        'profit': 'sum',
                        'category': lambda x: x.mode()[0] if not x.empty else None,
                        'sub-category': lambda x: ', '.join(x.dropna().unique()),
                        'quantity': 'sum' if 'quantity' in order_breakdown.columns else lambda x: None
                    }).reset_index()
                    merged_df = pd.merge(list_orders, order_breakdown_agg, on='order id', how='outer')
                    merged_df['profit'] = merged_df['profit'].fillna(0)
                    merged_df['sales'] = merged_df['sales'].fillna(0)
                    data_dict['Merged'] = merged_df
                except Exception as e:
                    logger.error(f"Failed to merge sheets: {str(e)}. Skipping merge.")

        for sheet_name, df in data_dict.items():
            logger.info(f"Analyzing sheet: {sheet_name}")
            df = df.copy()
            columns = detect_columns(df)

            if columns['sales']:
                df[columns['sales']] = pd.to_numeric(df[columns['sales']], errors='coerce').fillna(0)
            if columns['profit']:
                df[columns['profit']] = pd.to_numeric(df[columns['profit']], errors='coerce').fillna(0)

            info_buffer = io.StringIO()
            df.info(buf=info_buffer)
            info_str = info_buffer.getvalue()
            describe_str = custom_describe(df).to_string()
            corr_str = df.corr(numeric_only=True).to_string() if len(df.select_dtypes(include=['number']).columns) > 1 else 'Not available'

            total_sales = df[columns['sales']].sum() if columns['sales'] else None
            total_profit = df[columns['profit']].sum() if columns['profit'] else None
            profit_summary = None
            sales_summary = None
            region_sales_summary = None
            region_profit_summary = None
            category_counts = None
            main_category_counts = None
            profit_margins = None
            monthly_sales = None
            monthly_profit = None
            sales_profit_corr = None
            unique_customers = None

            if columns['sub_category'] and columns['category']:
                df.loc[df[columns['sub_category']].str.contains('Appliances', na=False), columns['category']] = 'Office Supplies'

            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            if numerical_cols.size > 0:
                col = numerical_cols[0]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                plt.title(f'Distribution of {col.capitalize()} ({sheet_name})')
                figures.append((fig, f"Histogram of {col} ({sheet_name})"))
                logger.info(f"Created histogram for {col} in sheet {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed histogram for {col} in sheet {sheet_name}")
            if categorical_cols.size > 0:
                col = categorical_cols[0]
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(y=col, data=df, order=df[col].value_counts().index[:10], ax=ax)
                plt.title(f'Count of {col.capitalize()} ({sheet_name})')
                figures.append((fig, f"Bar Chart of {col} ({sheet_name})"))
                logger.info(f"Created bar chart for {col} in sheet {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed bar chart for {col} in sheet {sheet_name}")

            if columns['sub_category'] and columns['profit']:
                profit_by_sub_category = df.groupby(columns['sub_category'])[columns['profit']].agg(['sum', 'count']).reset_index()
                profit_by_sub_category.columns = [columns['sub_category'], 'total_profit', 'order_count']
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x='total_profit', y=columns['sub_category'], data=profit_by_sub_category, ax=ax)
                plt.title(f'Total Profit by Sub-Category ({sheet_name})')
                plt.xlabel('Total Profit ($)')
                plt.ylabel('Sub-Category')
                figures.append((fig, f"Profit by Sub-Category ({sheet_name})"))
                logger.info(f"Created profit by sub-category plot for {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed profit by sub-category plot for {sheet_name}")
                profit_summary = profit_by_sub_category.to_string(index=False)
                profit_by_sub_category['average_profit'] = profit_by_sub_category['total_profit'] / profit_by_sub_category['order_count']
                category_counts = profit_by_sub_category[[columns['sub_category'], 'order_count', 'average_profit']].to_string(index=False)
            if columns['category'] and columns['profit'] and columns['sales']:
                profit_by_category = df.groupby(columns['category']).agg({
                    columns['profit']: ['sum', 'count'],
                    columns['sales']: 'sum'
                }).reset_index()
                profit_by_category.columns = [columns['category'], 'total_profit', 'order_count', 'total_sales']
                profit_by_category['average_profit'] = profit_by_category['total_profit'] / profit_by_category['order_count']
                profit_by_category['profit_margin'] = profit_by_category['total_profit'] / profit_by_category['total_sales']
                main_category_counts = profit_by_category[[columns['category'], 'order_count', 'average_profit', 'total_profit', 'total_sales']].to_string(index=False)
                profit_margins = profit_by_category[[columns['category'], 'profit_margin']].to_string(index=False)
            if columns['sub_category'] and columns['sales']:
                sales_by_sub_category = df.groupby(columns['sub_category'])[columns['sales']].sum().reset_index()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=columns['sales'], y=columns['sub_category'], data=sales_by_sub_category, ax=ax)
                plt.title(f'Total Sales by Sub-Category ({sheet_name})')
                plt.xlabel('Total Sales ($)')
                plt.ylabel('Sub-Category')
                figures.append((fig, f"Sales by Sub-Category ({sheet_name})"))
                logger.info(f"Created sales by sub-category plot for {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed sales by sub-category plot for {sheet_name}")
                sales_summary = sales_by_sub_category.to_string(index=False)
            if columns['region'] and columns['sales']:
                region_sales = df.groupby(columns['region'])[columns['sales']].sum().reset_index()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.barplot(x=columns['sales'], y=columns['region'], data=region_sales, ax=ax)
                plt.title(f'Total Sales by Region ({sheet_name})')
                plt.xlabel('Total Sales ($)')
                plt.ylabel('Region')
                figures.append((fig, f"Sales by Region ({sheet_name})"))
                logger.info(f"Created sales by region plot for {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed sales by region plot for {sheet_name}")
                region_sales_summary = region_sales.to_string(index=False)
            if columns['region'] and columns['profit']:
                region_profit = df.groupby(columns['region'])[columns['profit']].sum().reset_index()
                region_profit_summary = region_profit.to_string(index=False)
            if columns['sales'] and columns['profit']:
                sales_profit_corr = np.corrcoef(df[columns['sales']].dropna(), df[columns['profit']].dropna())[0, 1]
                sales_profit_corr = f"Correlation between sales and profit: {sales_profit_corr:.3f}"
            if columns['customer']:
                df[columns['customer']] = df[columns['customer']].str.strip().str.lower()
                unique_customers = df[columns['customer']].nunique()
            if columns['date'] and columns['sales']:
                df[columns['date']] = pd.to_datetime(df[columns['date']], errors='coerce', dayfirst=True)
                df['month_year'] = df[columns['date']].dt.to_period('M')
                monthly_sales = df.groupby('month_year')[columns['sales']].sum().reset_index()
                monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
                monthly_sales = monthly_sales.sort_values(columns['sales'], ascending=False).head(5)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(x='month_year', y=columns['sales'], data=monthly_sales, ax=ax)
                plt.title(f'Monthly Sales Trend ({sheet_name})')
                plt.xticks(rotation=45)
                figures.append((fig, f"Monthly Sales Trend ({sheet_name})"))
                logger.info(f"Created monthly sales trend plot for {sheet_name}")
                plt.close(fig)
                logger.info(f"Closed monthly sales trend plot for {sheet_name}")
                monthly_sales = monthly_sales.to_string(index=False)
                if columns['profit']:
                    monthly_profit = df.groupby('month_year')[columns['profit']].sum().reset_index()
                    monthly_profit['month_year'] = monthly_profit['month_year'].astype(str)
                    monthly_profit = monthly_profit.sort_values(columns['profit'], ascending=False).head(5)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.lineplot(x='month_year', y=columns['profit'], data=monthly_profit, ax=ax)
                    plt.title(f'Monthly Profit Trend ({sheet_name})')
                    plt.xticks(rotation=45)
                    figures.append((fig, f"Monthly Profit Trend ({sheet_name})"))
                    logger.info(f"Created monthly profit trend plot for {sheet_name}")
                    plt.close(fig)
                    logger.info(f"Closed monthly profit trend plot for {sheet_name}")
                    monthly_profit = monthly_profit.to_string(index=False)

            analysis_results[sheet_name] = {
                'columns': df.columns.tolist(),
                'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
                'nulls': df.isnull().sum().to_string(),
                'info': info_str,
                'describe': describe_str,
                'corr': corr_str,
                'total_sales': total_sales,
                'total_profit': total_profit,
                'profit_by_sub_category': profit_summary,
                'sales_by_sub_category': sales_summary,
                'region_sales': region_sales_summary,
                'region_profit': region_profit_summary,
                'sub_category_counts': category_counts,
                'main_category_counts': main_category_counts,
                'profit_margins': profit_margins,
                'monthly_sales': monthly_sales,
                'monthly_profit': monthly_profit,
                'sales_profit_corr': sales_profit_corr,
                'unique_customers': unique_customers,
                'detected_columns': columns
            }

        return analysis_results, figures
    except Exception as e:
        logger.error(f"Error analyzing DataFrames: {str(e)}")
        return None, []

# Question Answering with Fallback
def answer_question(context, question, retries=3, delay=10, timeout=60):
    prompt = f"""You are a data analyst expert. Answer the question accurately and concisely using the provided dataset context.

Dataset Context:
{context}

Question: {question}

Instructions:
- Use only summaries: total_sales, total_profit, unique_customers, main_category_counts, region_sales, monthly_sales, monthly_profit, sales_profit_corr.
- For totals, use 'total_sales' and 'total_profit' from 'Merged' sheet; fallback to 'OrderBreakdown'.
- For unique customers, use 'unique_customers' from 'ListOfOrders' or 'Merged'.
- For category questions (e.g., average profit by category), use 'main_category_counts' from 'OrderBreakdown' for raw order counts.
- For region sales (e.g., highest sales region), use 'region_sales' from 'Merged' sheet only.
- For monthly trends or highest profit month, use 'monthly_sales' and 'monthly_profit' from 'Merged' sheet only. List top 5 months for trends with values; exact month/value for highest profit.
- Ignore 'SalesTargets' for actual sales or profit.
- Appliances are part of Office Supplies, not Technology.
- If data is missing, state 'Data missing; check Merged sheet'.
- Do not guess or invent data.
- Format numbers with commas (e.g., 2,348,482) and round averages to 2 decimals.
"""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "grok-8b",  # Switch to a lighter model
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 512
    }
    for attempt in range(retries):
        try:
            logger.info(f"Attempting API call (Attempt {attempt+1}/{retries})...")
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            result = response.json()
            if "choices" in result and result["choices"]:
                return result["choices"][0]["message"]["content"].strip()
            else:
                logger.error("No choices in API response.")
                return "Error: No valid response from API."
        except requests.exceptions.Timeout:
            logger.error(f"API request timed out after {timeout} seconds.")
            st.error(f"API request timed out after {timeout} seconds. Retrying...")
            time.sleep(delay)
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                logger.warning(f"Rate limit error (429). Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                st.warning(f"Rate limit reached. Retrying in {delay} seconds... (Attempt {attempt+1}/{retries})")
                time.sleep(delay)
            else:
                logger.error(f"HTTP error querying Together AI: {str(e)}")
                return f"Error querying Together AI: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error querying Together AI: {str(e)}")
            return f"Unexpected error querying Together AI: {str(e)}"
    
    # Fallback response if API fails after retries
    fallback_response = (
        f"Unable to query the API after {retries} attempts due to timeout. "
        f"Here's a basic summary:\n"
        f"- Total Sales (Merged sheet): {context.split('Total Sales: ')[1].split('\n')[0] if 'Total Sales: ' in context else 'Not available'}\n"
        f"- Total Profit (Merged sheet): {context.split('Total Profit: ')[1].split('\n')[0] if 'Total Profit: ' in context else 'Not available'}"
    )
    return fallback_response

# Streamlit App
st.title("📊 Data Analyst App")
st.markdown("Upload a file (`.docx`, `.txt`, `.xlsx`, `.csv`, `.pdf`, `.png`) and ask questions about the data. Visualizations are generated for tabular files.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["docx", "txt", "xlsx", "csv", "pdf", "png"])

if uploaded_file is not None:
    st.success(f"File `{uploaded_file.name}` uploaded successfully!")
    try:
        # Load file
        content = load_file(uploaded_file)
        st.write("### File Content Preview")
        
        if isinstance(content, dict):  # Excel or CSV
            analysis_results, figures = analyze_dataframe(content)
            if analysis_results:
                context_parts = []
                for sheet_name, analysis in analysis_results.items():
                    total_sales_str = f"${analysis['total_sales']:,.2f}" if analysis['total_sales'] is not None else "Not available"
                    total_profit_str = f"${analysis['total_profit']:,.2f}" if analysis['total_profit'] is not None else "Not available"
                    context_parts.append(
                        f"\nSheet: {sheet_name}\n"
                        f"Total Sales: {total_sales_str}\n"
                        f"Total Profit: {total_profit_str}\n"
                        f"Unique Customers: {analysis['unique_customers'] if analysis['unique_customers'] is not None else 'Not available'}\n"
                        f"Sales by Region:\n{analysis['region_sales'] or 'Not available'}\n"
                        f"Main Category Counts:\n{analysis['main_category_counts'] or 'Not available'}\n"
                        f"Monthly Sales (Top 5):\n{analysis['monthly_sales'] or 'Not available'}\n"
                        f"Monthly Profit (Top 5):\n{analysis['monthly_profit'] or 'Not available'}\n"
                        f"Sales-Profit Correlation:\n{analysis['sales_profit_corr'] or 'Not available'}\n"
                    )
                    st.subheader(f"Sheet: {sheet_name}")
                    if sheet_name in content:
                        st.dataframe(content[sheet_name].head())
                context = "\n".join(context_parts)[:2000]
                st.write("### Analysis Summary")
                st.text(context[:500] + "..." if len(context) > 500 else context)
                
                # Display visualizations
                st.write("### Visualizations")
                if figures:
                    cols = st.columns(2)
                    for i, (fig, caption) in enumerate(figures):
                        with cols[i % 2]:
                            st.pyplot(fig)
                            st.caption(caption)
                            plt.close(fig)
                            logger.info(f"Closed figure after display: {caption}")
                else:
                    st.info("No visualizations generated for this file.")
                
                # Question input
                st.write("### Ask a Question")
                question = st.text_input("Enter your question about the data:", key="question_input")
                if st.button("Submit Question"):
                    if question:
                        with st.spinner("Processing question..."):
                            answer = answer_question(context, question)
                            st.write("#### Answer")
                            st.markdown(answer)
                    else:
                        st.warning("Please enter a question.")
            else:
                st.error("Analysis failed due to errors.")
                
        elif isinstance(content, str):  # Text, PDF, or PNG
            context = f"Textual content:\n{content[:2000]}"
            st.text_area("Extracted Text", content[:500] + "..." if len(content) > 500 else content, height=200)
            if uploaded_file.type == "image/png":
                uploaded_file.seek(0)  # Reset file pointer
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
                
            # Question input for text-based files
            st.write("### Ask a Question")
            question = st.text_input("Enter your question about the content:", key="question_input_text")
            if st.button("Submit Question", key="submit_text"):
                if question:
                    with st.spinner("Processing question..."):
                        answer = answer_question(context, question)
                        st.write("#### Answer")
                        st.markdown(answer)
                else:
                    st.warning("Please enter a question.")
                
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure the file is valid and dependencies are installed.")

else:
    st.info("Please upload a file to begin analysis.")

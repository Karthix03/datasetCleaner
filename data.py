import streamlit as st
import pandas as pd
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import io

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="🛡️ AI Dataset Cleaner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #155724;
    }
    
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        color: #856404;
    }
    
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .user-message {
        background: #007bff;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: right;
    }
    
    .bot-message {
        background: #e9ecef;
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        text-align: left;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API
@st.cache_resource
def init_genai():
    try:
        genai.configure(api_key="enter your api key here")
        return True
    except Exception as e:
        st.error(f"Error initializing Gemini AI: {str(e)}")
        return False

# ---------- Enhanced Functions ----------
@st.cache_data
def analyze_dataset(df):
    """Enhanced dataset analysis with more metrics"""
    total_entries = len(df)
    duplicate_entries = df.duplicated().sum()
    clean_entries = total_entries - duplicate_entries
    
    # Enhanced poisoned data detection
    poisoned_entries = df.isnull().sum().sum()
    
    # Check for suspicious patterns
    for col in df.select_dtypes(include=['object']):
        if col in df.columns:
            poisoned_entries += df[col].astype(str).str.contains(
                'hack|malware|virus|spam|phishing|attack', 
                case=False, 
                na=False
            ).sum()
    
    poisoned_percentage = (poisoned_entries / total_entries) * 100 if total_entries else 0
    duplicate_percentage = (duplicate_entries / total_entries) * 100 if total_entries else 0
    
    # Data type analysis
    numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
    text_cols = len(df.select_dtypes(include=['object']).columns)
    datetime_cols = len(df.select_dtypes(include=['datetime64']).columns)
    
    analysis = {
        "Total Entries": total_entries,
        "Total Columns": len(df.columns),
        "Duplicate Entries": int(duplicate_entries),
        "Duplicate %": round(duplicate_percentage, 2),
        "Poisoned Entries": int(poisoned_entries),
        "Poisoned %": round(poisoned_percentage, 2),
        "Clean Entries": int(clean_entries),
        "Numeric Columns": numeric_cols,
        "Text Columns": text_cols,
        "DateTime Columns": datetime_cols,
        "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
    }
    return analysis

def create_analysis_charts(df, analysis):
    """Create visualization charts for dataset analysis"""
    # Data Quality Chart
    fig1 = go.Figure(data=[
        go.Bar(x=['Clean', 'Duplicates', 'Poisoned'], 
               y=[analysis['Clean Entries'], analysis['Duplicate Entries'], analysis['Poisoned Entries']],
               marker_color=['#28a745', '#ffc107', '#dc3545'])
    ])
    fig1.update_layout(
        title="Data Quality Overview",
        xaxis_title="Data Type",
        yaxis_title="Count",
        height=400
    )
    
    # Missing Values Chart
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        fig2 = px.bar(
            x=missing_data.index,
            y=missing_data.values,
            title="Missing Values by Column",
            labels={'x': 'Columns', 'y': 'Missing Count'}
        )
        fig2.update_layout(height=400)
    else:
        fig2 = None
    
    # Data Types Distribution
    type_counts = {
        'Numeric': analysis['Numeric Columns'],
        'Text': analysis['Text Columns'],
        'DateTime': analysis['DateTime Columns']
    }
    
    fig3 = go.Figure(data=[go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=.3
    )])
    fig3.update_layout(title="Column Types Distribution", height=400)
    
    return fig1, fig2, fig3

@st.cache_data
def clean_dataset(df, remove_duplicates=True, remove_nulls=True, remove_poisoned=True):
    """Enhanced dataset cleaning with options"""
    df_cleaned = df.copy()
    cleaning_report = []
    
    original_shape = df_cleaned.shape
    
    if remove_duplicates:
        before_dup = len(df_cleaned)
        df_cleaned = df_cleaned.drop_duplicates()
        removed_dup = before_dup - len(df_cleaned)
        cleaning_report.append(f"Removed {removed_dup} duplicate rows")
    
    if remove_nulls:
        before_null = len(df_cleaned)
        df_cleaned = df_cleaned.dropna()
        removed_null = before_null - len(df_cleaned)
        cleaning_report.append(f"Removed {removed_null} rows with null values")
    
    if remove_poisoned:
        before_poison = len(df_cleaned)
        # Remove rows with suspicious content
        for col in df_cleaned.select_dtypes(include=['object']):
            if col in df_cleaned.columns:
                df_cleaned = df_cleaned[~df_cleaned[col].astype(str).str.contains(
                    'hack|malware|virus|spam|phishing|attack', 
                    case=False, 
                    na=False
                )]
        removed_poison = before_poison - len(df_cleaned)
        cleaning_report.append(f"Removed {removed_poison} rows with suspicious content")
    
    final_shape = df_cleaned.shape
    cleaning_report.append(f"Dataset shape changed from {original_shape} to {final_shape}")
    
    return df_cleaned, cleaning_report

def chatbot_response(user_input, df=None):
    """Enhanced chatbot response with better context"""
    if not init_genai():
        return "❌ AI service unavailable. Please check your API configuration."
    
    dataset_context = ""
    if df is not None:
        analysis = analyze_dataset(df)
        sample_data = df.head(2).to_dict(orient='records') if not df.empty else []
        
        dataset_context = f"""
        Current Dataset Context:
        - Columns: {list(df.columns)}
        - Shape: {df.shape[0]} rows × {df.shape[1]} columns
        - Data Quality: {analysis['Clean Entries']} clean, {analysis['Duplicate Entries']} duplicates, {analysis['Poisoned Entries']} poisoned
        - Column Types: {analysis['Numeric Columns']} numeric, {analysis['Text Columns']} text, {analysis['DateTime Columns']} datetime
        - Memory Usage: {analysis['Memory Usage (MB)']} MB
        - Sample Data: {sample_data}
        """

    prompt = f"""
    You are an expert AI dataset assistant. Analyze the user's request and provide helpful insights.

    {dataset_context}

    User Request: {user_input}

    Instructions:
    - Respond in the same language as the user
    - For dataset manipulation requests, explain what changes will be made
    - Provide actionable insights and recommendations
    - Be concise but informative
    - Use emojis appropriately for better readability
    """

    try:
        response = genai.GenerativeModel("gemini-2.5-flash").generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"

def apply_manipulation(df, user_input):
    """Enhanced dataset manipulation with more operations"""
    if df is None:
        return None, "⚠️ No dataset available for manipulation."

    user_input_lower = user_input.lower()
    df_updated = df.copy()
    
    try:
        # Delete column
        if any(word in user_input_lower for word in ["delete", "remove"]) and "column" in user_input_lower:
            for col in df.columns:
                if col.lower() in user_input_lower:
                    df_updated = df_updated.drop(columns=[col])
                    return df_updated, f"✅ Column '{col}' successfully removed."

        # Rename column
        if "rename" in user_input_lower and "to" in user_input_lower:
            parts = user_input_lower.replace("rename column", "").replace("rename", "").split("to")
            if len(parts) == 2:
                old_col, new_col = parts[0].strip(), parts[1].strip()
                for col in df.columns:
                    if col.lower() == old_col.lower():
                        df_updated = df_updated.rename(columns={col: new_col})
                        return df_updated, f"✅ Column '{col}' renamed to '{new_col}'."

        # Filter operations
        if "filter" in user_input_lower or "where" in user_input_lower:
            return df_updated, "🔍 Filtering operations detected. Please specify exact conditions for filtering."
        
        # Sort operations
        if "sort" in user_input_lower:
            for col in df.columns:
                if col.lower() in user_input_lower:
                    ascending = "desc" not in user_input_lower
                    df_updated = df_updated.sort_values(by=col, ascending=ascending)
                    direction = "ascending" if ascending else "descending"
                    return df_updated, f"📊 Dataset sorted by '{col}' in {direction} order."

        return df_updated, None

    except Exception as e:
        return df, f"❌ Could not apply manipulation: {str(e)}"

# ---------- Initialize Session State ----------
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_working' not in st.session_state:
    st.session_state.df_working = None

# ---------- Main UI ----------
# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ AI-Powered Dataset Cleaner & Assistant</h1>
    <p>Upload, analyze, clean, and interact with your datasets using advanced AI</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # File Upload Section
    st.markdown("### 📁 Dataset Upload")
    uploaded_file = st.file_uploader(
        "Choose your CSV file",
        type=["csv"],
        help="Upload a CSV file to start analyzing your data"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.df_working = df.copy()
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    
    # Dataset Actions
    if st.session_state.df is not None:
        st.markdown("### 🛠️ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Reset Data", use_container_width=True):
                st.session_state.df_working = st.session_state.df.copy()
                st.success("Data reset to original")
        
        with col2:
            if st.button("🧹 Auto Clean", use_container_width=True):
                cleaned_df, report = clean_dataset(st.session_state.df_working)
                st.session_state.df_working = cleaned_df
                st.success("Dataset auto-cleaned!")
        
        # Cleaning Options
        st.markdown("### ⚙️ Cleaning Options")
        remove_duplicates = st.checkbox("Remove Duplicates", value=True)
        remove_nulls = st.checkbox("Remove Null Values", value=True)
        remove_poisoned = st.checkbox("Remove Suspicious Data", value=True)
        
        if st.button("🎯 Custom Clean", use_container_width=True):
            cleaned_df, report = clean_dataset(
                st.session_state.df_working,
                remove_duplicates,
                remove_nulls,
                remove_poisoned
            )
            st.session_state.df_working = cleaned_df
            with st.expander("📋 Cleaning Report"):
                for item in report:
                    st.write(f"• {item}")

# Main Content Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "💬 AI Assistant",
    "🔧 Data Manipulation",
    "📥 Export & Reports"
])

# Tab 1: Dataset Overview
with tab1:
    if st.session_state.df_working is not None:
        df = st.session_state.df_working
        
        # Quick Stats
        st.subheader("📈 Quick Statistics")
        analysis = analyze_dataset(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{analysis['Total Entries']:,}")
        with col2:
            st.metric("Columns", analysis['Total Columns'])
        with col3:
            st.metric("Clean Data %", f"{100 - analysis['Duplicate %'] - analysis['Poisoned %']:.1f}%")
        with col4:
            st.metric("Memory (MB)", analysis['Memory Usage (MB)'])
        
        # Visualizations
        st.subheader("📊 Data Analysis Charts")
        fig1, fig2, fig3 = create_analysis_charts(df, analysis)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            if fig2:
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("🎉 No missing values found!")
        with col3:
            st.plotly_chart(fig3, use_container_width=True)
        
        # Dataset Preview
        st.subheader("👀 Dataset Preview")
        preview_rows = st.slider("Number of rows to preview", 5, 50, 10)
        st.dataframe(df.head(preview_rows), use_container_width=True)
        
        # Column Information
        with st.expander("📋 Column Details"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
    else:
        st.info("👈 Please upload a CSV file in the sidebar to start analyzing your data.")

# Tab 2: AI Assistant
with tab2:
    if st.session_state.df_working is not None:
        st.subheader("🤖 AI Dataset Assistant")
        
        # Chat Interface
        chat_container = st.container()
        
        # Display chat history
        if st.session_state.chat_history:
            with chat_container:
                for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
                    st.markdown(f"""
                    <div class="user-message">
                        👤 {user_msg}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="bot-message">
                        🤖 {bot_msg}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Input area
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input(
                "Ask about your data or request changes:",
                placeholder="e.g., 'What are the main patterns in this data?' or 'Delete the Age column'",
                key="chat_input"
            )
        with col2:
            send_button = st.button("Send 📤", use_container_width=True)
        
        if send_button and user_input:
            # Get AI response
            bot_response = chatbot_response(user_input, st.session_state.df_working)
            
            # Try to apply manipulation
            updated_df, manipulation_msg = apply_manipulation(st.session_state.df_working, user_input)
            
            if manipulation_msg:
                bot_response += f"\n\n**Action Taken:** {manipulation_msg}"
                if updated_df is not None:
                    st.session_state.df_working = updated_df
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, bot_response))
            st.rerun()
        
        # Quick suggestions
        st.subheader("💡 Quick Suggestions")
        suggestions = [
            "What patterns do you see in this data?",
            "Show me data quality issues",
            "Delete duplicate rows",
            "Sort by the first column",
            "What columns have missing values?"
        ]
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    bot_response = chatbot_response(suggestion, st.session_state.df_working)
                    st.session_state.chat_history.append((suggestion, bot_response))
                    st.rerun()
    else:
        st.info("👈 Please upload a dataset first to start chatting with the AI assistant.")

# Tab 3: Data Manipulation
with tab3:
    if st.session_state.df_working is not None:
        st.subheader("🔧 Advanced Data Manipulation")
        
        # Column Operations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Column Operations")
            selected_col = st.selectbox("Select Column", st.session_state.df_working.columns)
            
            operation = st.selectbox("Choose Operation", [
                "View Statistics",
                "Delete Column",
                "Rename Column",
                "Convert Data Type",
                "Fill Missing Values"
            ])
            
            if operation == "Delete Column":
                if st.button(f"Delete '{selected_col}'"):
                    st.session_state.df_working = st.session_state.df_working.drop(columns=[selected_col])
                    st.success(f"Column '{selected_col}' deleted!")
                    st.rerun()
            
            elif operation == "Rename Column":
                new_name = st.text_input("New column name:")
                if st.button("Rename") and new_name:
                    st.session_state.df_working = st.session_state.df_working.rename(columns={selected_col: new_name})
                    st.success(f"Column renamed to '{new_name}'!")
                    st.rerun()
            
            elif operation == "View Statistics":
                if st.session_state.df_working[selected_col].dtype in ['int64', 'float64']:
                    st.write(st.session_state.df_working[selected_col].describe())
                else:
                    st.write(f"Unique values: {st.session_state.df_working[selected_col].nunique()}")
                    st.write(f"Most common: {st.session_state.df_working[selected_col].mode().iloc[0] if not st.session_state.df_working[selected_col].mode().empty else 'N/A'}")
        
        with col2:
            st.markdown("#### Row Operations")
            
            # Filter rows
            if st.session_state.df_working.select_dtypes(include=[np.number]).columns.any():
                numeric_col = st.selectbox(
                    "Filter by numeric column",
                    st.session_state.df_working.select_dtypes(include=[np.number]).columns
                )
                
                min_val = float(st.session_state.df_working[numeric_col].min())
                max_val = float(st.session_state.df_working[numeric_col].max())
                
                range_values = st.slider(
                    f"Filter {numeric_col} range:",
                    min_val, max_val, (min_val, max_val)
                )
                
                if st.button("Apply Filter"):
                    mask = (st.session_state.df_working[numeric_col] >= range_values[0]) & \
                           (st.session_state.df_working[numeric_col] <= range_values[1])
                    st.session_state.df_working = st.session_state.df_working[mask]
                    st.success(f"Filtered data to {len(st.session_state.df_working)} rows")
                    st.rerun()
        
        # Current dataset status
        st.subheader("📊 Current Dataset Status")
        current_analysis = analyze_dataset(st.session_state.df_working)
        
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Rows", f"{current_analysis['Total Entries']:,}")
        with metrics_col2:
            st.metric("Columns", current_analysis['Total Columns'])
        with metrics_col3:
            st.metric("Data Quality", f"{100 - current_analysis['Poisoned %']:.1f}%")
        
    else:
        st.info("👈 Please upload a dataset first to start manipulating data.")

# Tab 4: Export & Reports
with tab4:
    if st.session_state.df_working is not None:
        st.subheader("📥 Export Options")
        
        # Generate comprehensive report
        analysis = analyze_dataset(st.session_state.df_working)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Dataset Summary Report")
            
            report = f"""
# Dataset Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
- **Total Entries**: {analysis['Total Entries']:,}
- **Total Columns**: {analysis['Total Columns']}
- **Data Quality Score**: {100 - analysis['Poisoned %']:.1f}%
- **Memory Usage**: {analysis['Memory Usage (MB)']} MB

## Data Quality Metrics
- **Clean Entries**: {analysis['Clean Entries']:,}
- **Duplicate Entries**: {analysis['Duplicate Entries']} ({analysis['Duplicate %']}%)
- **Suspicious Entries**: {analysis['Poisoned Entries']} ({analysis['Poisoned %']}%)

## Column Analysis
- **Numeric Columns**: {analysis['Numeric Columns']}
- **Text Columns**: {analysis['Text Columns']}
- **DateTime Columns**: {analysis['DateTime Columns']}

## Recommendations
- {'✅ Dataset quality is excellent!' if analysis['Poisoned %'] < 5 else '⚠️ Consider cleaning suspicious data'}
- {'✅ No duplicates found!' if analysis['Duplicate %'] == 0 else '🔄 Remove duplicate entries'}
- {'💾 Consider data compression for large datasets' if analysis['Memory Usage (MB)'] > 100 else '💾 Memory usage is optimal'}
            """
            
            st.markdown(report)
        
        with col2:
            st.markdown("#### 💾 Download Options")
            
            # CSV Download
            csv_buffer = io.StringIO()
            st.session_state.df_working.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                "📄 Download CSV",
                csv_data,
                f"cleaned_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            # Excel Download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                st.session_state.df_working.to_excel(writer, sheet_name='Cleaned_Data', index=False)
                
                # Add analysis sheet
                analysis_df = pd.DataFrame(list(analysis.items()), columns=['Metric', 'Value'])
                analysis_df.to_excel(writer, sheet_name='Analysis', index=False)
            
            st.download_button(
                "📊 Download Excel (with Analysis)",
                excel_buffer.getvalue(),
                f"dataset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Report Download
            st.download_button(
                "📋 Download Report (Markdown)",
                report,
                f"dataset_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown",
                use_container_width=True
            )
        
        # Comparison with original
        if st.session_state.df is not None and not st.session_state.df.equals(st.session_state.df_working):
            st.subheader("🔄 Changes Summary")
            
            original_analysis = analyze_dataset(st.session_state.df)
            current_analysis = analyze_dataset(st.session_state.df_working)
            
            comparison_data = {
                'Metric': ['Total Rows', 'Total Columns', 'Duplicate %', 'Poisoned %', 'Memory (MB)'],
                'Original': [
                    original_analysis['Total Entries'],
                    original_analysis['Total Columns'],
                    original_analysis['Duplicate %'],
                    original_analysis['Poisoned %'],
                    original_analysis['Memory Usage (MB)']
                ],
                'Current': [
                    current_analysis['Total Entries'],
                    current_analysis['Total Columns'],
                    current_analysis['Duplicate %'],
                    current_analysis['Poisoned %'],
                    current_analysis['Memory Usage (MB)']
                ]
            }
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df['Change'] = comparison_df['Current'] - comparison_df['Original']
            
            st.dataframe(comparison_df, use_container_width=True)
    
    else:
        st.info("👈 Please upload and process a dataset first to access export options.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🛡️ AI Dataset Cleaner | Built with Streamlit & Google Gemini | "
    f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</div>", 
    unsafe_allow_html=True
)
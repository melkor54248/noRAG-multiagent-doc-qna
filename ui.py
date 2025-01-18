import streamlit as st
from io import BytesIO
from utils import extract_text_from_pdf, process_document_chunks
from document_selection import select_relevant_document, get_answer

# Page configuration
st.set_page_config(
    page_title="PDF Document Q&A System",
    page_icon="📚",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 50%;
        }
        .upload-text {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .status-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .token-info {
            font-size: 0.9rem;
            color: #666;
            padding: 5px;
            border-radius: 5px;
            background-color: #f0f2f6;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for documents and UI control
if 'documents' not in st.session_state:
    st.session_state.documents = {}
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'token_counts' not in st.session_state:
    st.session_state.token_counts = {}
if 'show_answer' not in st.session_state:
    st.session_state.show_answer = False

st.subheader("📚 Multiagent Document QnA")

# Create tabs
tab1, tab2 = st.tabs(["Main", "Configuration"])

with tab1:
    # Main UI
    col1, col2 = st.columns([3, 2])

    with col1:
        # Main content area
        st.markdown("#### 📄 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",  # Changed from empty string
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF files to analyze",
            key="pdf_uploader",
            label_visibility="collapsed"  # Hides the label but maintains accessibility
        )

        # Process uploaded files
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.documents:
                    with st.spinner('🔄 Document Analysis Agent is processing document ' + file.name):
                        try:
                            progress_bar = st.progress(0)
                            
                            progress_bar.progress(25)
                            chunks, chunk_tokens = extract_text_from_pdf(BytesIO(file.read()))
                            
                            progress_bar.progress(50)
                            total_tokens = sum(chunk_tokens)
                            
                            if len(chunks) > 1:
                                st.info(f"""
                                    ℹ️ Document '{file.name}' is large ({total_tokens:,} tokens) and will be split into {len(chunks)} parts.
                                    Each part will be processed separately for better handling.
                                """)
                            
                            progress_bar.progress(75)
                            docs, sums, tokens = process_document_chunks(file.name, chunks, chunk_tokens)
                            
                            st.session_state.documents.update(docs)
                            st.session_state.summaries.update(sums)
                            st.session_state.token_counts.update(tokens)
                            
                            progress_bar.progress(100)
                            
                            if len(chunks) > 1:
                                st.success(f"""
                                    ✅ Successfully processed {file.name}
                                    \n📊 Total tokens: {total_tokens:,}
                                    \n📑 Split into {len(chunks)} parts of {', '.join(f"{tokens:,}" for tokens in chunk_tokens)} tokens each
                                """)
                            else:
                                st.success(f"""
                                    ✅ Successfully processed {file.name}
                                    \n📊 Token count: {total_tokens:,} tokens
                                """)
                            
                            progress_bar.empty()
                            
                        except Exception as e:
                            st.error(f"""
                                ❌ Error processing {file.name}
                                \nError: {str(e)}
                                \nPlease try again with a different file or contact support if the issue persists.
                            """)
                            continue

        st.markdown("#### ❓ Ask Your Question")
        question = st.text_input(
            "Enter your question",  # Changed from empty string
            key="question_input",
            placeholder="Type your question here...",
            help="Ask a question about the uploaded documents",
            label_visibility="collapsed"  # Hides the label but maintains accessibility
        )
        
        if st.button("🔍 Submit Question", type="primary"):
            st.session_state.show_answer = True
        else:
            st.session_state.show_answer = False

        if st.session_state.show_answer and question and st.session_state.documents:
            with st.spinner('🔍 Researcher Agent is analyzing document relevance...'):
                relevant_doc, relevance_scores = select_relevant_document(question, st.session_state.summaries)
                
                st.markdown("#### 📊 Document Relevance")
                
                sorted_scores = dict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
                
                with st.expander("View Relevance Scores"):
                    for doc, score in sorted_scores.items():
                        col_0, col_1, col_2 = st.columns([3, 2, 0.5])
                        with col_0:
                            st.markdown(f"{doc}")
                        with col_1:
                            st.progress(score / 100)
                        with col_2:
                            st.markdown(f"{score}%")

            with st.spinner('🔍 Reply Agent is generating an answer from the most relevant document...'):
                answer = get_answer(question, st.session_state.documents[relevant_doc])
                
                st.markdown("#### 💡 Answer")
                st.info(f"""
                    📄 Source: {relevant_doc}
                    \n📊 Document size: {st.session_state.token_counts[relevant_doc]:,} tokens
                    \n🎯 Relevance score: {relevance_scores[relevant_doc]}%
                """)
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
                        {answer}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    with col2:
        st.markdown("#### 📑 Documents Processed")
        
        if st.session_state.summaries:
            total_tokens = sum(st.session_state.token_counts.values())
            st.markdown(f"📊 Total tokens across all documents: **{total_tokens:,}**")
            
            for filename in st.session_state.summaries.keys():
                with st.expander(f"📄 {filename}"):
                    # Make summary editable with automatic saving
                    edited_summary = st.text_area(
                        "Document Appendix",
                        value=st.session_state.summaries[filename],
                        height=350,
                        key=f"summary_{filename}",
                        help="Edit this summary to refine document matching"
                    )
                    
                    # Update the summary if changed
                    if edited_summary != st.session_state.summaries[filename]:
                        st.session_state.summaries[filename] = edited_summary
                    
                    st.markdown(
                        f"""
                        <div class="token-info">
                            📊 Number of Tokens in Document: {st.session_state.token_counts[filename]:,}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    

        else:
            st.info("📌 Upload documents to see their summaries here")

        st.markdown("#### 🔧 System Status")
        with st.expander("View Details", expanded=True):
            st.markdown(f"**Documents Loaded:** {len(st.session_state.documents)}")
            st.markdown(f"**Model:** {deployment_name}")
            if st.session_state.token_counts:
                st.markdown(f"**Total Tokens:** {sum(st.session_state.token_counts.values()):,}")
            st.markdown("**Status:** 🟢 System Ready")

with tab2:
    st.markdown("#### ⚙️ System Configuration")

    # Document Processing Settings
    with st.expander("📄 Document Processing Settings"):
        processing_config = st.session_state.config.get_processing_config()
        new_max_tokens = st.number_input(
            "Maximum Tokens per Chunk",
            min_value=1000,
            max_value=200000,
            value=processing_config['max_chunk_tokens'],
            help="Maximum number of tokens per document chunk",
            key="doc_proc_max_tokens"
        )
        if new_max_tokens != processing_config['max_chunk_tokens']:
            st.session_state.config.update_config('document_processing', 'max_chunk_tokens', new_max_tokens)

    # Document Analysis Agent Configuration
    with st.expander("📊 Document Analysis Agent"):
        doc_analysis_config = st.session_state.config.get_agent_config('document_analysis_agent')

        new_system_prompt = st.text_area(
            "System Prompt",
            value=doc_analysis_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role",
            key="doc_analysis_system_prompt"
        )
        if new_system_prompt != doc_analysis_config['system_prompt']:
            st.session_state.config.update_config('document_analysis_agent', 'system_prompt', new_system_prompt)

        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=doc_analysis_config['model_prompt'],
            height=100,
            help="Template for the model prompt (actual text will be appended)",
            key="doc_analysis_model_prompt"
        )
        if new_model_prompt != doc_analysis_config['model_prompt']:
            st.session_state.config.update_config('document_analysis_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=2000,
                value=doc_analysis_config['max_tokens'],
                help="Maximum number of tokens for response",
                key="doc_analysis_max_tokens"
            )
            if new_max_tokens != doc_analysis_config['max_tokens']:
                st.session_state.config.update_config('document_analysis_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=doc_analysis_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative)",
                key="doc_analysis_temperature"
            )
            if new_temperature != doc_analysis_config['temperature']:
                st.session_state.config.update_config('document_analysis_agent', 'temperature', new_temperature)

    # Researcher Agent Configuration
    with st.expander("🔍 Researcher Agent"):
        researcher_config = st.session_state.config.get_agent_config('researcher_agent')
        
        new_system_prompt = st.text_area(
            "System Prompt",
            value=researcher_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role",
            key="researcher_system_prompt"
        )
        if new_system_prompt != researcher_config['system_prompt']:
            st.session_state.config.update_config('researcher_agent', 'system_prompt', new_system_prompt)
        
        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=researcher_config['model_prompt'],
            height=100,
            help="Template for the model prompt (document details will be appended)",
            key="researcher_model_prompt"
        )
        if new_model_prompt != researcher_config['model_prompt']:
            st.session_state.config.update_config('researcher_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=2000,
                value=researcher_config['max_tokens'],
                help="Maximum number of tokens for response",
                key="researcher_max_tokens"
            )
            if new_max_tokens != researcher_config['max_tokens']:
                st.session_state.config.update_config('researcher_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=researcher_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative)",
                key="researcher_temperature"
            )
            if new_temperature != researcher_config['temperature']:
                st.session_state.config.update_config('researcher_agent', 'temperature', new_temperature)

    # Reply Agent Configuration
    with st.expander("💡 Reply Agent"):
        reply_config = st.session_state.config.get_agent_config('reply_agent')
        
        new_system_prompt = st.text_area(
            "System Prompt",
            value=reply_config['system_prompt'],
            height=100,
            help="System prompt that defines the agent's role",
            key="reply_system_prompt"
        )
        if new_system_prompt != reply_config['system_prompt']:
            st.session_state.config.update_config('reply_agent', 'system_prompt', new_system_prompt)
        
        new_model_prompt = st.text_area(
            "Model Prompt Template",
            value=reply_config['model_prompt'],
            height=100,
            help="Template for the model prompt (document context will be appended)",
            key="reply_model_prompt"
        )
        if new_model_prompt != reply_config['model_prompt']:
            st.session_state.config.update_config('reply_agent', 'model_prompt', new_model_prompt)
        
        col1, col2 = st.columns(2)
        with col1:
            new_max_tokens = st.number_input(
                "Maximum Tokens",
                min_value=100,
                max_value=2000,
                value=reply_config['max_tokens'],
                help="Maximum number of tokens for response",
                key="reply_max_tokens"
            )
            if new_max_tokens != reply_config['max_tokens']:
                st.session_state.config.update_config('reply_agent', 'max_tokens', new_max_tokens)
        
        with col2:
            new_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=reply_config['temperature'],
                step=0.1,
                help="Controls randomness in generation (0 = deterministic, 1 = creative)",
                key="reply_temperature"
            )
            if new_temperature != reply_config['temperature']:
                st.session_state.config.update_config('reply_agent', 'temperature', new_temperature)


    # Model Information
    st.markdown("#### 🤖 Model Information")
    azure_config = st.session_state.config.get_azure_config()
    st.info(f"""
        **Current Model:** {azure_config['deployment_name']}
        \n**API Version:** {azure_config['api_version']}
        \n**Endpoint:** {azure_config['azure_endpoint']}
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        Made using Streamlit and Azure OpenAI
    </div>
    """,
    unsafe_allow_html=True
)

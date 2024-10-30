import streamlit as st
import PyPDF2
import os
from io import BytesIO
from openai import AzureOpenAI
import tiktoken
import json
from typing import List, Dict, Tuple
import logging

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

# Configure OpenAI
client = AzureOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv('OPENAI_ENDPOINT')
)
deployment_name = os.getenv('OPENAI_DEPLOYMENT_NAME')

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(encoding.encode(text))

def split_text_into_chunks(text: str, max_tokens: int = 120000) -> List[str]:
    """Split text into chunks of maximum token size."""
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        if current_length >= max_tokens:
            # Decode current chunk into text
            chunk_text = encoding.decode(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
        
        current_chunk.append(token)
        current_length += 1
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = encoding.decode(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def extract_text_from_pdf(pdf_file) -> Tuple[List[str], List[int]]:
    """Extract text from a PDF file and return text chunks and their token counts."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    
    # Count total tokens
    total_tokens = count_tokens(full_text)
    
    # If document is too large, split it into chunks
    if total_tokens > 120000:
        chunks = split_text_into_chunks(full_text)
        chunk_tokens = [count_tokens(chunk) for chunk in chunks]
        return chunks, chunk_tokens
    else:
        return [full_text], [total_tokens]

def get_summary(text: str) -> str:
    """Get summary of text using OpenAI."""
    prompt = f"Please provide a concise summary of the following text:\n\n{text}"
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that creates an appendix of the document. This appendix will later be used to match the document to questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=300
    )
    
    return response.choices[0].message.content

def process_document_chunks(file_name: str, chunks: List[str], chunk_tokens: List[int]) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, int]]:
    """Process multiple chunks of a document and return their data."""
    documents = {}
    summaries = {}
    token_counts = {}
    
    for i, (chunk, tokens) in enumerate(zip(chunks, chunk_tokens)):
        if len(chunks) > 1:
            chunk_name = f"{file_name} (Part {i+1}/{len(chunks)})"
        else:
            chunk_name = file_name
            
        documents[chunk_name] = chunk
        token_counts[chunk_name] = tokens
        
        # Get summary for the chunk
        summary = get_summary(chunk)
        summaries[chunk_name] = summary
    
    return documents, summaries, token_counts

def select_relevant_document(question: str, summaries: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    """
    Select the most relevant document based on the question and summaries.
    Returns the most relevant document filename and a dictionary of relevance scores.
    """
    prompt = """Given the following document appendices and a question, analyze each document's relevance to the question.
    Return a JSON object with filename keys and relevance scores (0-100) as values.
    Only return the JSON object, no other text.
    
    Example format:
    {
        "document1.pdf": 85,
        "document2.pdf": 45
    }
    
    Documents and summaries:\n\n"""
    
    for filename, summary in summaries.items():
        prompt += f"Document: {filename}\nSummary: {summary}\n\n"
    
    prompt += f"Question: {question}\n\nRelevance scores:"
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates document relevance. Respond only with a JSON object containing filename keys and relevance score values (0-100). Don't use ```json or ```, just return the pure JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    try:
        # Parse the response into a dictionary
        logging.info(response.choices[0].message.content)
        relevance_scores = json.loads(response.choices[0].message.content)
        # Find the most relevant document (highest score)
        most_relevant = max(relevance_scores.items(), key=lambda x: x[1])[0]
        return most_relevant, relevance_scores
    except json.JSONDecodeError:
        st.error("Error parsing relevance scores. Using fallback method.")
        return list(summaries.keys())[0], {k: 0 for k in summaries.keys()}

def get_answer(question: str, document_text: str) -> str:
    """Get answer to question using the selected document."""
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Use ONLY the following document to answer questions. DO NOT MAKE UP ANY INFO. If the answer is not within the document say that you don't know. Document for Context:\n\n{document_text}"},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

# Main UI
st.subheader("📚 Multiagent Information Retrieval")
st.markdown("---")

# Create two columns for layout
col1, col2 = st.columns([3, 2])

with col1:
    # Main content area
    st.markdown("#### 📄 Upload Documents")
    
    # Initialize session state
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'summaries' not in st.session_state:
        st.session_state.summaries = {}
    if 'token_counts' not in st.session_state:
        st.session_state.token_counts = {}
    if 'show_answer' not in st.session_state:
        st.session_state.show_answer = False

    # File uploader with enhanced styling
    uploaded_files = st.file_uploader(
        "",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF files to analyze"
    )

    # Process uploaded files
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.documents:
                with st.spinner('🔄 Document Analysis Agent is processing document ' + file.name):
                    try:
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        
                        # Extract text and count tokens
                        progress_bar.progress(25)
                        chunks, chunk_tokens = extract_text_from_pdf(BytesIO(file.read()))
                        
                        # Process chunks
                        progress_bar.progress(50)
                        total_tokens = sum(chunk_tokens)
                        
                        if len(chunks) > 1:
                            st.info(f"""
                                ℹ️ Document '{file.name}' is large ({total_tokens:,} tokens) and will be split into {len(chunks)} parts.
                                Each part will be processed separately for better handling.
                            """)
                        
                        # Process each chunk
                        progress_bar.progress(75)
                        docs, sums, tokens = process_document_chunks(file.name, chunks, chunk_tokens)
                        
                        # Update session state
                        st.session_state.documents.update(docs)
                        st.session_state.summaries.update(sums)
                        st.session_state.token_counts.update(tokens)
                        
                        progress_bar.progress(100)
                        
                        # Show success message
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

    # Question input and submit button
    st.markdown("#### ❓ Ask Your Question")
    question = st.text_input(
        "",
        key="question_input",
        placeholder="Type your question here...",
        help="Ask a question about the uploaded documents"
    )
    
    if st.button("🔍 Submit Question", type="primary"):
        st.session_state.show_answer = True
    else:
        st.session_state.show_answer = False

    if st.session_state.show_answer and question and st.session_state.documents:
        with st.spinner('🔍 Researcher Agent is analyzing document relevance...'):
            # Select relevant document and get relevance scores
            relevant_doc, relevance_scores = select_relevant_document(question, st.session_state.summaries)
            
            # Display relevance scores
            st.markdown("#### 📊 Document Relevance")
            
            # Sort documents by relevance score
            sorted_scores = dict(sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True))
            
            # Display relevance scores in a more compact format
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
            # Get answer
            answer = get_answer(question, st.session_state.documents[relevant_doc])
            
            # Display results in a nice formatted box
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
    # Sidebar content
    st.markdown("#### 📑 Documents Processed")
    
    if st.session_state.summaries:
        total_tokens = sum(st.session_state.token_counts.values())
        st.markdown(f"📊 Total tokens across all documents: **{total_tokens:,}**")
        
        for filename, summary in st.session_state.summaries.items():
            with st.expander(f"📄 {filename}"):
                st.markdown(
                    f"""
                    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
                        {summary}
                    </div>
                    <div class="token-info">
                        📊 Tokens: {st.session_state.token_counts[filename]:,}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.info("📌 Upload documents to see their summaries here")

    # System Status
    st.markdown("#### 🔧 System Status")
    with st.expander("View Details", expanded=True):
        st.markdown(f"**Documents Loaded:** {len(st.session_state.documents)}")
        st.markdown(f"**Model:** {deployment_name}")
        if st.session_state.token_counts:
            st.markdown(f"**Total Tokens:** {sum(st.session_state.token_counts.values()):,}")
        st.markdown("**Status:** 🟢 System Ready")

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
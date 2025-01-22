import tiktoken
from io import BytesIO
from openai import AzureOpenAI
import json
from typing import List, Dict, Tuple
import logging
import streamlit as st
import PyPDF2
from configuration.config import ConfigLoader
from mm_doc_proc.multimodal_processing_pipeline.configuration_models import ProcessingPipelineConfiguration
from mm_doc_proc.multimodal_processing_pipeline.pdf_ingestion_pipeline import PDFIngestionPipeline
from mm_doc_proc.multimodal_processing_pipeline.data_models import DocumentContent
from mm_doc_proc.utils.openai_data_models import (
    MulitmodalProcessingModelInfo, 
    TextProcessingModelnfo
)

# Initialize configuration
if 'config' not in st.session_state:
    st.session_state.config = ConfigLoader()

# Configure OpenAI
azure_config = st.session_state.config.get_azure_config()
client = AzureOpenAI(
    api_key=azure_config['api_key'],
    api_version=azure_config['api_version'],
    azure_endpoint=azure_config['azure_endpoint']
)
deployment_name = azure_config['deployment_name']

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(encoding.encode(text))

def extract_text_from_pdf_gpt(pdf_file) -> Tuple[List[str], List[int]]:
    """Extract text from a PDF file using multimodal processing pipeline."""
    # Create pipeline configuration
    pipeline_config = ProcessingPipelineConfiguration(
        pdf_path=pdf_file,
        output_directory= 'tmp/',
        process_text=True,  # Enable text processing
        process_images=True,  # Enable image processing
        process_tables=True,  # Enable table processing
        save_text_files=True,
        generate_condensed_text=False,  # We'll handle summarization separately
        generate_table_of_contents=False
    )
    
    # Configure models using existing Azure configuration
    pipeline_config.text_model = TextProcessingModelnfo(
        provider="azure",
        model_name='o1',
        reasoning_efforts="medium",
        endpoint=azure_config['azure_endpoint'],
        key=azure_config['api_key'],
        model=deployment_name,
        api_version=azure_config['api_version']
    )
    
    pipeline_config.multimodal_model = MulitmodalProcessingModelInfo(
        provider="azure",
        model_name='o1',
        reasoning_efforts="medium",
        endpoint=azure_config['azure_endpoint'],
        key=azure_config['api_key'],
        model=deployment_name,
        api_version=azure_config['api_version']
    )
    
    # Initialize and run pipeline
    pipeline = PDFIngestionPipeline(pipeline_config)
    document_content: DocumentContent = pipeline.process_pdf()
    document_text = document_content.full_text
    total_tokens = count_tokens(document_text)
    max_chunk_tokens = st.session_state.config.get_processing_config()['max_chunk_tokens']
        
    #     # Split into chunks if necessary
    if total_tokens > max_chunk_tokens:
        chunks = split_text_into_chunks(document_text)
        chunk_tokens = [count_tokens(chunk) for chunk in chunks]
        return chunks, chunk_tokens
    else:
        return [document_text], [total_tokens]

def extract_text_from_pdf_pypdf2(pdf_file) -> Tuple[List[str], List[int]]:
    """Extract text from a PDF file and return text chunks and their token counts."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    full_text = ""
    for page in pdf_reader.pages:
        full_text += page.extract_text()
    
    total_tokens = count_tokens(full_text)
    max_chunk_tokens = st.session_state.config.get_processing_config()['max_chunk_tokens']
    
    if total_tokens > max_chunk_tokens:
        chunks = split_text_into_chunks(full_text)
        chunk_tokens = [count_tokens(chunk) for chunk in chunks]
        return chunks, chunk_tokens
    else:
        return [full_text], [total_tokens]


def split_text_into_chunks(text: str, max_tokens: int = None) -> List[str]:
    """Split text into chunks of maximum token size."""
    if max_tokens is None:
        max_tokens = st.session_state.config.get_processing_config()['max_chunk_tokens']
        
    tokens = encoding.encode(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for token in tokens:
        if current_length >= max_tokens:
            chunk_text = encoding.decode(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_length = 0
        
        current_chunk.append(token)
        current_length += 1
    
    if current_chunk:
        chunk_text = encoding.decode(current_chunk)
        chunks.append(chunk_text)
    
    return chunks

def get_summary(text: str) -> str:
    """Get summary of text using OpenAI."""
    config = st.session_state.config.get_agent_config('document_analysis_agent')
    prompt = config['model_prompt'] + text
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": prompt}
        ],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
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
        
        summary = get_summary(chunk)
        summaries[chunk_name] = summary
    
    return documents, summaries, token_counts

def select_relevant_document(question: str, summaries: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    """Select the most relevant document based on the question and summaries."""
    config = st.session_state.config.get_agent_config('researcher_agent')
    prompt = config['model_prompt'] + "\n\nDocuments and summaries:\n\n"
    
    for filename, summary in summaries.items():
        prompt += f"Document: {filename}\nSummary: {summary}\n\n"
    
    prompt += f"Question: {question}\n\nRelevance scores:"
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": prompt}
        ],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
    )

    try:
        logging.info(response.choices[0].message.content)
        relevance_scores = json.loads(response.choices[0].message.content)
        most_relevant = max(relevance_scores.items(), key=lambda x: x[1])[0]
        return most_relevant, relevance_scores
    except json.JSONDecodeError:
        st.error("Error parsing relevance scores. Using fallback method.")
        return list(summaries.keys())[0], {k: 0 for k in summaries.keys()}

def get_answer(question: str, document_text: str) -> str:
    """Get answer to question using the selected document."""
    config = st.session_state.config.get_agent_config('reply_agent')
    prompt = config['model_prompt'] + question
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": config['system_prompt'] + "\n\nDocument Context:\n" + document_text},
            {"role": "user", "content": prompt}
        ],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
    )
    
    return response.choices[0].message.content

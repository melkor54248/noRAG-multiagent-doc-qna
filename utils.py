from typing import List, Dict, Tuple
import tiktoken
import PyPDF2
from io import BytesIO
import json
import logging
from openai import AzureOpenAI
import streamlit as st

# Initialize tokenizer
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    return len(encoding.encode(text))

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

def extract_text_from_pdf(pdf_file) -> Tuple[List[str], List[int]]:
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

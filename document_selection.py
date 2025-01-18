from typing import Dict, Tuple
from openai import AzureOpenAI
import streamlit as st

def select_relevant_document(question: str, summaries: Dict[str, str]) -> Tuple[str, Dict[str, float]]:
    """Select the most relevant document based on the question."""
    config = st.session_state.config.get_agent_config('researcher_agent')
    prompt = config['model_prompt'] + json.dumps(summaries) + "\n\nQuestion: " + question
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": config['system_prompt']},
            {"role": "user", "content": prompt}
        ],
        temperature=config['temperature'],
        max_tokens=config['max_tokens']
    )
    
    relevance_scores = json.loads(response.choices[0].message.content)
    relevant_doc = max(relevance_scores, key=relevance_scores.get)
    
    return relevant_doc, relevance_scores

def get_answer(question: str, document: str) -> str:
    """Get answer to the question based on the document."""
    config = st.session_state.config.get_agent_config('reply_agent')
    prompt = config['model_prompt'] + document + "\n\nQuestion: " + question
    
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

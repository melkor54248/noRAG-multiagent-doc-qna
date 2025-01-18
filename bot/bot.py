import os
import json
from typing import List, Dict, Tuple
from botbuilder.core import ActivityHandler, MessageFactory, TurnContext
from botbuilder.schema import ChannelAccount, Attachment
from botbuilder.core.integration import BotFrameworkAdapter, BotFrameworkHttpClient
from botbuilder.core.integration.async_middleware import Middleware
from botbuilder.core.integration.turn_context import TurnContext
from utils import extract_text_from_pdf, process_document_chunks, select_relevant_document, get_answer

class DocumentQnABot(ActivityHandler):
    def __init__(self):
        self.documents = {}
        self.summaries = {}
        self.token_counts = {}

    async def on_message_activity(self, turn_context: TurnContext):
        text = turn_context.activity.text.strip().lower()
        if text.startswith("upload"):
            await self.handle_file_upload(turn_context)
        elif text.startswith("question"):
            await self.handle_question(turn_context)
        else:
            await turn_context.send_activity("Please upload a file or ask a question.")

    async def handle_file_upload(self, turn_context: TurnContext):
        for attachment in turn_context.activity.attachments:
            if attachment.content_type == "application/pdf":
                file_name = attachment.name
                file_content = await self.download_attachment(attachment.content_url)
                chunks, chunk_tokens = extract_text_from_pdf(BytesIO(file_content))
                docs, sums, tokens = process_document_chunks(file_name, chunks, chunk_tokens)
                self.documents.update(docs)
                self.summaries.update(sums)
                self.token_counts.update(tokens)
                await turn_context.send_activity(f"Successfully processed {file_name}")

    async def handle_question(self, turn_context: TurnContext):
        question = turn_context.activity.text[len("question"):].strip()
        if not self.documents:
            await turn_context.send_activity("No documents uploaded.")
            return
        relevant_doc, relevance_scores = select_relevant_document(question, self.summaries)
        answer = get_answer(question, self.documents[relevant_doc])
        await turn_context.send_activity(f"Answer: {answer}")

    async def download_attachment(self, content_url: str) -> bytes:
        # Implement the logic to download the file from the content URL
        pass

# Refactor repetitive code into reusable functions to avoid redundancy
def create_bot():
    return DocumentQnABot()

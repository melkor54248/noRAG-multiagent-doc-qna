import os
from aiohttp import web
from botbuilder.core import (
    BotFrameworkAdapterSettings,
    TurnContext,
    BotFrameworkAdapter,
    ConversationState,
    MemoryStorage,
)
from botbuilder.schema import Activity, ActivityTypes
from services.document_service import (
    count_tokens,
    split_text,
    extract_text,
    summarize,
    process_chunks,
    select_relevant,
    get_answer,
)

# Bot configuration
APP_ID = os.getenv("MICROSOFT_APP_ID", "")
APP_PASSWORD = os.getenv("MICROSOFT_APP_PASSWORD", "")
adapter_settings = BotFrameworkAdapterSettings(APP_ID, APP_PASSWORD)
adapter = BotFrameworkAdapter(adapter_settings)

# Memory storage for conversation state
memory_storage = MemoryStorage()
conversation_state = ConversationState(memory_storage)


async def handle_message(context: TurnContext):
    if context.activity.type == ActivityTypes.message:
        user_input = context.activity.text

        if user_input.startswith("upload"):
            # Handle document upload
            file_data = await context.activity.attachments[0].download()
            extraction_result = await extract_text(file_data)
            await context.send_activity("Document uploaded and processed successfully.")
            await context.send_activity(f"Extracted text: {extraction_result['chunks'][:500]}...")

        elif user_input.startswith("question"):
            # Handle question submission
            question = user_input[len("question "):]
            summaries = {}  # Fetch summaries from your storage
            relevant_doc = await select_relevant(question, summaries)
            answer = await get_answer(question, relevant_doc["most_relevant"])
            await context.send_activity(f"Answer: {answer}")

        else:
            await context.send_activity("Please upload a document or ask a question.")

    await conversation_state.save_changes(context)


async def messages(req: web.Request) -> web.Response:
    body = await req.json()
    activity = Activity().deserialize(body)
    auth_header = req.headers.get("Authorization", "")

    response = await adapter.process_activity(activity, auth_header, handle_message)
    return web.json_response(data=response.body, status=response.status)


app = web.Application()
app.router.add_post("/api/messages", messages)

if __name__ == "__main__":
    web.run_app(app, host="localhost", port=3978)

import httpx

API_BASE_URL = "http://localhost:8000"

async def count_tokens(text: str) -> int:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/count_tokens/", json={"text": text})
        response.raise_for_status()
        return response.json()["token_count"]

async def split_text(text: str) -> list:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/split_text/", json={"text": text})
        response.raise_for_status()
        return response.json()["chunks"]

async def extract_text(file: bytes) -> dict:
    async with httpx.AsyncClient() as client:
        files = {"file": ("document.pdf", file, "application/pdf")}
        response = await client.post(f"{API_BASE_URL}/extract_text/", files=files)
        response.raise_for_status()
        return response.json()

async def summarize(text: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/summarize/", json={"text": text})
        response.raise_for_status()
        return response.json()["summary"]

async def process_chunks(file_name: str, chunks: list, chunk_tokens: list) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/process_chunks/", json={"file_name": file_name, "chunks": chunks, "chunk_tokens": chunk_tokens})
        response.raise_for_status()
        return response.json()

async def select_relevant(question: str, summaries: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/select_relevant/", json={"question": question, "summaries": summaries})
        response.raise_for_status()
        return response.json()

async def get_answer(question: str, document_text: str) -> str:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{API_BASE_URL}/get_answer/", json={"question": question, "document_text": document_text})
        response.raise_for_status()
        return response.json()["answer"]

import os
import json
import getpass
import logging
import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

llm = ChatOllama(model="llama3.2", temperature=0.9)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can answer questions about the world. I will ask you a question and you will provide an answer. If you don't know the answer, you can say 'I don't know'.",
        ),
        ("human", "{input}"),
    ]
)

chain = prompt | llm


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(title="Ollama API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/search")
def search(query: str):
    ai_msg = chain.invoke(query)
    return ai_msg.content


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            for chunk in chain.stream(data):
                await websocket.send_text(chunk.content)
    except WebSocketDisconnect:
        logging.info("WebSocket connection closed")
        await websocket.close()


async def event_generator(data):
    try:
        for chunk in chain.stream(data):
            response_str = "data: " + json.dumps(chunk.content) + "\n\n"
            yield response_str.encode("utf-8")
        # logging.info("Event generator done")
        # end
        yield "data: {}\n\n"
    except Exception as e:
        logging.error(str(e))
        yield f"data: {str(e)}\n\n".encode("utf-8")


@app.get("/sse")
async def sse(query: str):
    # logging.info(f"Query: {query}")
    return StreamingResponse(event_generator(query), media_type="text/event-stream")


@app.get("/", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

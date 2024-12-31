import os
import getpass
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
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


app = FastAPI(title="Ollama API")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.post("/search")
def search(query: str):
    ai_msg = chain.invoke(query)
    return ai_msg.content


@app.get("/", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

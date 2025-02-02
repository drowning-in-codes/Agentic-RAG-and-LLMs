from operator import itemgetter
from typing import List, Literal

import faiss
import gradio as gr
from diffusers import DDPMPipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from typing_extensions import TypedDict


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


store = {}


def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]


default_prompt = ChatPromptTemplate(
    [
        (
            "system",
            "You are a helpful assistant that can answer questions about the world. I will ask you a question and you will provide an answer. If you don't know the answer, you can say 'I don't know'."
            "use the following context to help answering: {context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
physics_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.""",
        ),
        ("human", "{question}"),
    ]
)

math_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts, 
answer the component parts, and then put them together\
to answer the broader question.""",
        ),
        ("human", "{question}"),
    ]
)

history_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.""",
        ),
        ("human", "{question}"),
    ]
)

computerscience_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity.""",
        ),
        ("human", "{question}"),
    ]
)


class RouteQuery(TypedDict):
    """Route query to destination."""

    destination: Literal["Math", "Physics", "History", "Computer Science", "Default"]


route_system = """Route the user's query to the appropriate assistant. 
 The assistant will then provide an answer to the user's query. The output format will be 
{{destination: string}} 
context information: {{context}}.
The destination can be one of the following: Math, Physics, History, Computer Science, and Default. 
"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_system),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

chain_with_history = None
route_chain = None
default_chain = None
math_chain = None
physics_chain = None
history_chain = None
computerscience_chain = None


def select_chain(chain_name):
    destination = chain_name["destination"]
    if destination == "Math":
        return math_chain
    elif destination == "Physics":
        return physics_chain
    elif destination == "History":
        return history_chain
    elif destination == "Computer Science":
        return computerscience_chain
    else:
        return default_chain


auto_route_chain = None


def prepare_model(model_name):
    global chain_with_history, route_chain, default_chain, physics_chain, math_chain, history_chain, computerscience_chain, auto_route_chain
    if model_name.lower() == "deepseek":
        llm = ChatOllama(model="deepseek", temperature=0.9)
    else:
        llm = ChatOllama(model="llama3.2", temperature=0.9)
    math_chain = math_template | llm | StrOutputParser()
    physics_chain = physics_template | llm | StrOutputParser()
    history_chain = history_template | llm | StrOutputParser()
    computerscience_chain = computerscience_template | llm | StrOutputParser()
    default_chain = default_prompt | llm | StrOutputParser()
    route_chain = (
        route_prompt
        | llm.with_structured_output(RouteQuery)
        | itemgetter("destination")
    )

    chain_with_history = RunnableWithMessageHistory(
        default_chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )
    auto_route_chain = {
        "destination": route_chain,
        "question": lambda x: x["question"],
        "history": lambda x: x["history"],
    } | RunnableLambda(select_chain)


text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained("thenlper/gte-small"),
    chunk_size=200,
    chunk_overlap=20,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)


embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello AI")))
vectordb = FAISS(
    embedding_function=embedding_model,
    distance_strategy=DistanceStrategy.COSINE,
    index=index,
    index_to_docstore_id={},
    docstore=InMemoryDocstore(),
)

docs = []


async def talk(history: list):
    global docs, chain_with_history
    context = None
    user_msg = history[-1]["content"]
    if len(docs) != 0:
        context = vectordb.search(user_msg, search_type="similarity")
    history.append({"role": "assistant", "content": ""})
    for chunk in chain_with_history.stream(
        {"question": user_msg, "history": history, "context": context},
        config={"configurable": {"session_id": "pro"}},
    ):
        if isinstance(chunk, str):
            history[-1]["content"] += chunk
        elif isinstance(chunk, dict):
            history[-1]["content"] += chunk["content"]
        yield history


def user(model, user_message, history: list):
    prepare_model(model)
    return "", history + [{"role": "user", "content": user_message}]


ddpm = None


def generate_images(prompt: str):
    global ddpm
    if ddpm is None:
        download_models
        ddpm = DDPMPipeline.from_pretrained(
            "google/ddpm-cat-256", use_safetensors=True
        ).to("cuda")
    image = ddpm(num_inference_steps=25).images[0]
    return image


def download_models():
    global ddpm
    gr.Info("Downloading models...")
    ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to(
        "cuda"
    )


def add_file(loaded_file):
    global docs
    if loaded_file:
        docs.append(loaded_file.name)
        pages = []
        for page in PyPDFLoader(loaded_file).lazy_load():
            pages.append(page)
        doc = text_splitter.split_documents(pages)
        vectordb.add_documents(doc)
        gr.Info("File uploaded successfully")
    else:
        gr.Info("No file uploaded")


def autoRoute(enable_route: bool):
    global chain_with_history, auto_route_chain
    if enable_route:
        chain_with_history = RunnableWithMessageHistory(
            auto_route_chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )
    else:
        chain_with_history = RunnableWithMessageHistory(
            default_chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )


with gr.Blocks() as app:
    gr.Markdown("## Langchain Chat")
    model = gr.Dropdown(
        choices=["llama3.2", "deepseek"], label="model", info="Choose your model!"
    )
    loaded_file = gr.File(label="Upload pdf File", file_types=[".pdf"])
    chatbot = gr.Chatbot(type="messages")
    with gr.Group():
        route_btn = gr.Checkbox(label="Enable Auto Router", value=False)
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    msg.submit(
        user,
        [
            model,
            msg,
            chatbot,
        ],
        [msg, chatbot],
    ).then(talk, [chatbot], chatbot)

    route_btn.change(autoRoute, route_btn)
    clear.click(lambda: None, None, chatbot)
    loaded_file.change(add_file, loaded_file)

    with gr.Group():
        gr.Label("Generate Images")
        download_model = gr.Button("check whether downloads models or not")
        prompt = gr.Textbox(label="Prompt")
        generate = gr.Button("Generate")
        image = gr.Image()
    download_model.click(download_models)
    generate.click(generate_images, prompt, image)


if __name__ == "__main__":
    app.launch()

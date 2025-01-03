from operator import itemgetter
from typing import List, Literal
from datetime import date
import faiss
import gradio as gr
import langchain
from langchain.agents.agent import RunnableAgent
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
from langchain_community.agent_toolkits.load_tools import load_tools
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from typing_extensions import TypedDict
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)
from langchain.agents import AgentExecutor
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.tools.convert import tool


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

llm = ChatOllama(model="llama3.2", temperature=0.9)

math_chain = math_template | llm | StrOutputParser()
physics_chain = physics_template | llm | StrOutputParser()
history_chain = history_template | llm | StrOutputParser()
computerscience_chain = computerscience_template | llm | StrOutputParser()


@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())


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
default_chain = default_prompt | llm | StrOutputParser()
chain_with_history = RunnableWithMessageHistory(
    default_chain,
    get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
).with_config({"configurable": {"session_id": "pro"}})
tools = load_tools(["wikipedia"])
print({tool.name: tool for tool in tools})

agent = RunnableAgent(
    runnable=chain_with_history,
    input_keys_arg=["question", "context"],
    output_keys_arg=["content"],
)
agent_executor = AgentExecutor(agent=agent,tools=tools+[PythonREPLTool(),time], verbose=True)
route_chain = (
    route_prompt | llm.with_structured_output(RouteQuery) | itemgetter("destination")
)


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


auto_route_chain = {
    "destination": route_chain,
    "question": lambda x: x["question"],
    "history": lambda x: x["history"],
} | RunnableLambda(select_chain)


def autoRoute(enable_route: bool):
    global chain_with_history, agent_executor
    if enable_route:
        chain_with_history = RunnableWithMessageHistory(
            auto_route_chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )

        chain_with_history.config = {"configurable": {"session_id": "pro"}}
        agent = RunnableAgent(
            runnable=chain_with_history,
            input_keys_arg=["question", "context"],
            output_keys_arg=["content"],
        )
        agent_executor.agent = agent

    else:
        chain_with_history = RunnableWithMessageHistory(
            default_chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history",
        )
        chain_with_history.config = {"configurable": {"session_id": "pro"}}
        agent = RunnableAgent(
            runnable=chain_with_history,
            input_keys_arg=["question"],
            output_keys_arg=["content"],
        )

        agent_executor.agent = agent


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
    global docs
    context = None
    user_msg = history[-1]["content"]
    if len(docs) != 0:
        context = vectordb.search(user_msg, search_type="similarity")
    history.append({"role": "assistant", "content": ""})
    for chunk in agent_executor.stream(
        {"question": user_msg, "history": history, "context": context},
        config={"configurable": {"session_id": "pro"}},
    ):
        if isinstance(chunk, str):
            history[-1]["content"] += chunk
        elif isinstance(chunk, dict):
            history[-1]["content"] += chunk["content"]
        yield history


def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


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


def eval_langchain(eval_btn):
    if eval_btn:
        langchain.debug = True
    else:
        langchain.debug = False


with gr.Blocks() as app:
    gr.Markdown("## Langchain Chat")
    loaded_file = gr.File(label="Upload pdf File", file_types=[".pdf"])
    chatbot = gr.Chatbot(type="messages")
    with gr.Group():
        route_btn = gr.Checkbox(label="Enable Auto Router", value=False)
        eval_btn = gr.Checkbox(label="Enable Evaluation", value=False)
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(
        user,
        [
            msg,
            chatbot,
        ],
        [msg, chatbot],
    ).then(talk, [chatbot], chatbot)
    route_btn.change(autoRoute, route_btn)
    eval_btn.change(eval_langchain, eval_btn)
    clear.click(lambda: None, None, chatbot)
    loaded_file.change(add_file, loaded_file)


if __name__ == "__main__":
    app.launch()

from operator import itemgetter
from typing import List
import gradio as gr
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, AIMessage
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Literal
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter


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
            "You are a helpful assistant that can answer questions about the world. I will ask you a question and you will provide an answer. If you don't know the answer, you can say 'I don't know'.",
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
that you don't know.

Here is a question:
{question}""",
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


class RouteQuery(TypedDict):
    """Route query to destination."""

    destination: Literal["Math", "Physics", "History", "Computer Science", "Default"]


route_system = """Route the user's query to the appropriate assistant. 
 The assistant will then provide an answer to the user's query. The output format will be 
{{destination: string}} 
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
)

route_chain = (
    route_prompt | llm.with_structured_output(RouteQuery) | itemgetter("destination")
)


def select_chain(chain_name):
    print(chain_name)
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
    "destination": route_chain,  # "animal" or "vegetable"
    "question": lambda x: x["question"],  # pass through input query
    "history": lambda x: x["history"],  # pass through history
} | RunnableLambda(select_chain)


def autoRoute(enable_route: bool):
    global chain_with_history
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


def talk(history: list):
    user_msg = history[-1]["content"]
    history.append({"role": "assistant", "content": ""})
    for chunk in chain_with_history.stream(
        {"question": user_msg, "history": history},
        config={"configurable": {"session_id": "pro"}},
    ):
        if isinstance(chunk, str):
            history[-1]["content"] += chunk
        elif isinstance(chunk, dict):
            history[-1]["content"] += chunk["content"]
        yield history


def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


with gr.Blocks() as app:
    gr.Markdown("## Langchain Chat")
    chatbot = gr.Chatbot(type="messages")
    with gr.Group():
        route_btn = gr.Checkbox(label="Enable Auto Router", value=False)
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(talk, [chatbot], chatbot)
    route_btn.change(autoRoute, route_btn)
    clear.click(lambda: None, None, chatbot)


if __name__ == "__main__":
    app.launch()

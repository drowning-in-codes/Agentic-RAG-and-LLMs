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


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that can answer questions about the world. I will ask you a question and you will provide an answer. If you don't know the answer, you can say 'I don't know'.",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{question}"),
    ]
)
llm = ChatOllama(model="llama3.2", temperature=0.9)

chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
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
        history[-1]["content"] += chunk.content
        yield history


def user(user_message, history: list):
    return "", history + [{"role": "user", "content": user_message}]


with gr.Blocks() as app:
    gr.Markdown("## Langchain Chat")
    chatbot = gr.Chatbot(type="messages")
    msg = gr.Textbox()
    clear = gr.Button("Clear")

    msg.submit(user, [msg, chatbot], [msg, chatbot]).then(talk, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot)


if __name__ == "__main__":
    app.launch()

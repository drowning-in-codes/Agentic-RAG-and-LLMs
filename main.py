from enum import Enum
import os
import getpass
import gradio as gr
from gradio import ChatMessage
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, ToolCollection
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from dataclasses import dataclass
from gradio.components import Component
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore
from transformers.agents import HfApiEngine, ReactJsonAgent, load_tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from transformers.agents import PythonInterpreterTool


class RetrieverTool(Tool):
    name = "retriever"
    description = "Using semantic similarity, retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, vectordb: VectorStore, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.vectordb.similarity_search(
            query,
            k=7,
        )

        gr.Info(
            "\nRetrieved documents:\n"
            + "".join(
                [
                    f"===== Document {str(i)} =====\n" + doc.page_content
                    for i, doc in enumerate(docs)
                ]
            )
        )


async def process_files(files, docs):
    for file in files:
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file)
            async for page in loader.alazy_load():
                docs.append(page)
        elif file.endswith(".csv"):
            loader = CSVLoader(file)
            docs.extend(loader.load())
        elif file.endswith(".html"):
            loader = UnstructuredHTMLLoader(file)
            docs.extend(loader.load())
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file)
            data = loader.load()
            docs.extend(data)
        else:
            raise gr.Error("Unsupported file type")


async def process_file_query(query, history, *config):

    rag_config = RAGConfig.fromList(config)
    docs = []
    process_files(rag_config.files, docs)
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in docs
    ]
    if rag_config.tokenizer_source == Provider.OpenAI:
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=rag_config.tokenizer,
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(rag_config.tokenizer),
            chunk_size=200,
            chunk_overlap=20,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
    docs_processed = []
    unique_texts = {}
    for doc in tqdm(source_docs):
        new_docs = text_splitter.split_documents([doc])
        for new_doc in new_docs:
            if new_doc.page_content not in unique_texts:
                unique_texts[new_doc.page_content] = True
                docs_processed.append(new_doc)
    if rag_config.embedding_source == Provider.OpenAI:
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        from langchain_openai import OpenAIEmbeddings

        embedding_model = OpenAIEmbeddings(model_name=rag_config.embedding_model)
    else:
        embedding_model = HuggingFaceEmbeddings(model_name=rag_config.embedding_model)
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=rag_config.search_strategy,
    )
    llm_engine = HfApiEngine(rag_config.infer_model)
    retriever_tool = RetrieverTool(vectordb)
    agent = ReactJsonAgent(
        tools=[retriever_tool, PythonInterpreterTool()],
        add_base_tool=True,
        additional_authorized_imports=["requests", "bs4"],
        llm_engine=llm_engine,
        max_iterations=4,
        verbose=2,
    )
    history.append(ChatMessage(role="user", content=query))
    agent_output = agent.run(query)
    history.append(ChatMessage(role="assistant", content=agent_output))
    return query, history, ""


@dataclass
class RAGConfig:
    tokenizer: Component
    tokenizer_source: Component
    embedding_model: Component
    embedding_source: Component
    search_strategy: Component
    infer_model: Component
    files: Component

    def toList(self):
        return [
            self.tokenizer,
            self.tokenizer_source,
            self.embedding_model,
            self.embedding_source,
            self.search_strategy,
            self.infer_model,
            self.files,
        ]

    @staticmethod
    def fromList(l):
        return RAGConfig(*l)


class DistanceStrategy(Enum):
    COSINE = "COSINE"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"


class Provider(Enum):
    HuggingFace = "Hugging Face"
    OpenAI = "OpenAI"


# 定义模型选项
TOKENIZER_GROUP = {
    Provider.HuggingFace: ["thenlper/gte-small", "bert-base-uncased"],
    Provider.OpenAI: ["cl100k_base", "gpt-2"],
}
EMBEDDING_GROUP = {
    Provider.HuggingFace: ["thenlper/gte-small", "bert-base-uncased"],
    Provider.OpenAI: ["gpt-2"],
}

INFER_MODEL = [
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "deepseek-ai/DeepSeek-V3-Base",
    "meta-llama/Llama-3.2-1B",
    "facebook/blenderbot-400M-distill",
    "deepset/roberta-base-squad2",
    "FlagAlpha/Llama2-Chinese-7b-Chat",
]


def switch_model_options(source):
    return gr.Dropdown(
        choices=EMBEDDING_GROUP[source], value=EMBEDDING_GROUP[source][0]
    )


def switch_tokenizer_options(source):
    return gr.Dropdown(
        choices=TOKENIZER_GROUP[source], value=TOKENIZER_GROUP[source][0]
    )


def update_embedding_options(source, embedding_source):
    if source in EMBEDDING_GROUP[embedding_source]:
        return gr.Dropdown(
            choices=TOKENIZER_GROUP[embedding_source][source], value=source
        )
    else:
        EMBEDDING_GROUP[embedding_source].append(source)
        return gr.Dropdown(choices=EMBEDDING_GROUP[embedding_source], value=source)


def update_tokenizer_options(source, tokenizer_source):
    if source in TOKENIZER_GROUP[tokenizer_source]:
        return gr.Dropdown(
            choices=TOKENIZER_GROUP[tokenizer_source][source], value=source
        )
    else:
        TOKENIZER_GROUP[tokenizer_source].append(source)
        return gr.Dropdown(choices=TOKENIZER_GROUP[tokenizer_source], value=source)


ALLOW_FILE_TYPES = [".pdf", ".csv", ".html", ".md"]

with gr.Blocks() as app:
    stored_message = gr.State([])
    with gr.Group():
        local_files = gr.File(
            label="上传文件", file_count="multiple", file_types=ALLOW_FILE_TYPES
        )
        with gr.Row():
            with gr.Column():
                # 选择模型来源
                tokenizer_source = gr.Radio(
                    choices=[Provider.HuggingFace.value, Provider.OpenAI.value],
                    label="tokenizer来源",
                    value=Provider.HuggingFace,  # 默认选中 Hugging Face
                )
                tokenizer = gr.Dropdown(
                    label="Tokenizer",
                    choices=TOKENIZER_GROUP[Provider.HuggingFace],
                    value=TOKENIZER_GROUP[Provider.HuggingFace][0],
                )
                custom_token = gr.Textbox(label="自定义Tokenizer")
                custom_token.submit(
                    update_tokenizer_options,
                    inputs=[custom_token, tokenizer_source],
                    outputs=[tokenizer],
                )

            with gr.Column():
                # 选择模型来源
                embedding_source = gr.Radio(
                    choices=[Provider.HuggingFace.value, Provider.OpenAI.value],
                    label="embedding来源",
                    value=Provider.HuggingFace,  # 默认选中 Hugging Face
                )
                embedding_model = gr.Dropdown(
                    label="embedding model",
                    choices=EMBEDDING_GROUP[Provider.HuggingFace],
                    value=EMBEDDING_GROUP[Provider.HuggingFace][0],
                )
                custom_embedding = gr.Textbox(label="自定义Embedding")
                custom_embedding.submit(
                    update_embedding_options,
                    inputs=[custom_embedding, embedding_source],
                    outputs=[embedding_model],
                )
        search_strategy = gr.Dropdown(
            label="search strategy",
            choices=[
                DistanceStrategy.COSINE.value,
                DistanceStrategy.DOT_PRODUCT.value,
                DistanceStrategy.JACCARD.value,
                DistanceStrategy.MAX_INNER_PRODUCT.value,
                DistanceStrategy.EUCLIDEAN_DISTANCE.value,
            ],
            value=DistanceStrategy.COSINE,
        )

        tokenizer_source.change(
            fn=switch_tokenizer_options, inputs=tokenizer_source, outputs=tokenizer
        )
        embedding_source.change(
            fn=switch_model_options, inputs=embedding_source, outputs=embedding_model
        )
        infer_model = gr.Dropdown(
            choices=INFER_MODEL, label="Inference Model", value=INFER_MODEL[0]
        )
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="你要问些什么?")
        clear = gr.ClearButton([msg, chatbot])
        rag_config = RAGConfig(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            search_strategy=search_strategy,
            embedding_source=embedding_source,
            tokenizer_source=tokenizer_source,
            files=local_files,
            infer_model=infer_model,
        )
        msg.submit(
            process_file_query,
            inputs=[msg, chatbot, *rag_config.toList()],
            outputs=[stored_message, chatbot, msg],
        )


if __name__ == "__main__":
    app.launch()

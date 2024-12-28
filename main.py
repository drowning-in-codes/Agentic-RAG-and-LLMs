from enum import Enum
import gradio as gr
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm
from dataclasses import dataclass
from gradio.components import Component


async def process_file_query(file, query, *rag_config, history):
    print(file_type)
    if file_type == "pdf":
        loader = PyPDFLoader(file)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    source_docs = [
        Document(page_content=doc["text"], metadata={"source": doc["source"]})
        for doc in pages
    ]
    if rag_config.tokenizer_source == "Open":
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder()
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

    embedding_model = HuggingFaceEmbeddings(model_name=rag_config.embedding_model)
    vectordb = FAISS.from_documents(
        documents=docs_processed,
        embedding=embedding_model,
        distance_strategy=rag_config.search_strategy,
    )


def interact_with_agent(prompt, messages):
    pass


@dataclass
class RAGConfig:
    tokenizer: Component
    tokenizer_source: Component
    embedding_model: Component
    embedding_source: Component
    file_type: Component
    search_strategy: Component

    def toList(self):
        return [
            self.tokenizer,
            self.embedding_model,
            self.file_type,
            self.search_strategy,
        ]

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

def switch_model_options(source):
    return gr.Dropdown(choices=EMBEDDING_GROUP[source], value=EMBEDDING_GROUP[source][0])

def switch_tokenizer_options(source):
    return gr.Dropdown(choices=TOKENIZER_GROUP[source], value=TOKENIZER_GROUP[source][0])

def update_embedding_options(source,embedding_source):
    if source in EMBEDDING_GROUP[embedding_source]:
        return gr.Dropdown(
            choices=TOKENIZER_GROUP[embedding_source][source], value=source
        )
    else:
        EMBEDDING_GROUP[embedding_source].append(source)
        return gr.Dropdown(choices=EMBEDDING_GROUP[embedding_source], value=source)

def update_tokenizer_options(source,tokenizer_source):
    if source in TOKENIZER_GROUP[tokenizer_source]:
        return gr.Dropdown(
            choices=TOKENIZER_GROUP[tokenizer_source][source], value=source
        )
    else:
        TOKENIZER_GROUP[tokenizer_source].append(source)
        return gr.Dropdown(choices=TOKENIZER_GROUP[tokenizer_source], value=source)


with gr.Blocks() as app:
    with gr.Group():
        with gr.Column():
            local_file = gr.File(label="上传文件")
            file_type = gr.Dropdown(
                label="文件类型",
                choices=["pdf", "webpages", "csv", "txt", "docx", "html", "json"],
                value="pdf",
            )
        with gr.Row():
            with gr.Column():
                # 选择模型来源
                tokenizer_source = gr.Radio(
                    choices=["Hugging Face", "OpenAI"],
                    label="tokenizer来源",
                    value="Hugging Face",  # 默认选中 Hugging Face
                )
                tokenizer = gr.Dropdown(
                    label="Tokenizer",
                    choices=TOKENIZER_GROUP["Hugging Face"],
                    value=TOKENIZER_GROUP["Hugging Face"][0],
                )
                custom_token = gr.Textbox(label="自定义Tokenizer")
                custom_token.submit(update_tokenizer_options, inputs=[custom_token,tokenizer_source], outputs=[tokenizer])

            with gr.Column():
                # 选择模型来源
                embedding_source = gr.Radio(
                    choices=["Hugging Face", "OpenAI"],
                    label="embedding来源",
                    value="Hugging Face",  # 默认选中 Hugging Face
                )
                embedding_model = gr.Dropdown(
                    label="embedding model",
                    choices=EMBEDDING_GROUP["Hugging Face"],
                    value=EMBEDDING_GROUP["Hugging Face"][0],
                )
                custom_embedding = gr.Textbox(label="自定义Embedding")
                custom_embedding.submit(update_embedding_options, inputs=[custom_embedding,embedding_source], outputs=[embedding_model])
        search_strategy = gr.Dropdown(
            label="search strategy",
            choices=[
                DistanceStrategy.COSINE,
                DistanceStrategy.DOT_PRODUCT,
                DistanceStrategy.JACCARD,
                DistanceStrategy.MAX_INNER_PRODUCT,
                DistanceStrategy.EUCLIDEAN_DISTANCE,
            ],
            value=DistanceStrategy.COSINE,
        )

        rag_config = RAGConfig(
            file_type=file_type,
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            search_strategy=search_strategy,
        )
        tokenizer_source.change(
            fn=switch_tokenizer_options, inputs=tokenizer_source, outputs=tokenizer
        )
        embedding_source.change(
            fn=switch_model_options, inputs=embedding_source, outputs=embedding_model
        )
        chatbot = gr.Chatbot(type="messages")
        msg = gr.Textbox(placeholder="你要问些什么?")
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(
            process_file_query,
            inputs=[local_file, msg, *rag_config.toList(), chatbot],
            outputs=[msg, chatbot],
        ).then(interact_with_agent, inputs=[msg, chatbot], outputs=[chatbot])


if __name__ == "__main__":
    app.launch()

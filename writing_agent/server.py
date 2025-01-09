import json
import io
import os
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sanic import Sanic
from sanic.response import text
from sanic import response
from diffusers import DiffusionPipeline
from sanic_cors import CORS, cross_origin
import torch


from sanic.exceptions import SanicException, BadRequest
from cors import add_cors_headers
from options import setup_options

app = Sanic("Write-AI")
BASE_URL = "/api/v1"

cors = CORS(
    app,
    resources={f"{BASE_URL}/*": {"origins": "*"}},
    automatic_options=True,
)


@app.get("/")
async def hello_world(request):
    return text("Hello, world.")


class FileReceiver:
    def __init__(self, request) -> None:
        self.file_dir = "./files"
        self.file_name = request.get("filename", None)
        if self.file_name == None:
            raise SanicException("file_name is None", BadRequest)
        self.totalChunks = request.get("totalChunks", 0)
        if self.totalChunks == 0:
            raise SanicException("total chunks is 0", BadRequest)
        self.metadata = request.get("metadata")
        self.file_data = b""
        self.chunk_count = 0

    def receive_chunks(self, message):
        self.file_data += message
        self.chunk_count += 1

    def end(self):
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)
        if self.chunk_count != self.totalChunks:
            raise SanicException("chunk count is not equal to total chunks", BadRequest)
        try:
            with open(self.file_path, "wb") as f:
                f.write(self.file_data)
        except Exception as e:
            print(e)

    @property
    def receive_finish(self):
        return self.chunk_count == self.totalChunks

    @property
    def file_path(self):
        return os.path.join(self.file_dir, self.file_name)


@app.post(f"{BASE_URL}/edraw", name="image")
async def image(request):
    prompt = request.json["prompt"] or "An image of a squirrel in Picasso style"
    pipeline = DiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
    )
    pipeline.to("cuda")
    img = pipeline(prompt).images[0]
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr = img_byte_arr.getvalue()
    # with open("./imgs/user.png", "rb") as f:
    #     img_byte_arr = f.read()
    return response.raw(img_byte_arr, content_type="image/png")


@app.get(f"{BASE_URL}/ewrite", name="write")
async def write(request):
    llm = ChatOllama(model="llama3.2")
    messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage("As a writer, write according to the following prompt"),
            HumanMessagePromptTemplate.from_template(
                "{query},the writing style is {style},the language is {language}"
            ),
        ]
    )
    inputMsg = request.args.get("input_msg", "")
    writingStyle = request.args.get("style", "normal")
    language = request.args.get("language", "Chinese")
    headers = {"Cache-Control": "no-cache"}
    response = await request.respond(headers=headers, content_type="text/event-stream")
    while True:
        for chunk in llm.stream(
            messages.format_messages(
                query=inputMsg, style=writingStyle, language=language
            )
        ):
            print(chunk.content)
            await response.send(f"data: {chunk.content}\n\n")
        await response.send(f"data: close\n\n")


@app.get(f"{BASE_URL}/etranslate", name="translate")
async def translate(request):
    llm = ChatOllama(model="llama3.2")
    messages = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                "As a translator,translate the following message, default to english"
            ),
            HumanMessagePromptTemplate.from_template(
                "Translate the following text {query} to English, just output the answer.\n\n"
            ),
        ]
    )
    inputMsg = request.args.get("input_msg", "")
    headers = {"Cache-Control": "no-cache"}
    response = await request.respond(headers=headers, content_type="text/event-stream")
    while True:
        for chunk in llm.stream(messages.format_messages(query=inputMsg)):
            print(chunk.content)
            await response.send(f"data: {chunk.content}\n\n")
        await response.send(f"data: close\n\n")


@app.websocket(f"{BASE_URL}/translate", name="ws_translate")
async def translate(request, ws):
    file_receiver = None
    prompt = None
    template_messages = [
        SystemMessage(
            "translate the following message according to the query, default to english"
        ),
        HumanMessagePromptTemplate.from_template(
            "Translate the following text according to the query {query} and if no text is provided, please translate the query to English.\n\n"
        ),
    ]
    try:
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                if file_receiver is None:
                    await ws.send(
                        json.dumps(
                            {"type": "error", "message": "No file transfer initiated"}
                        )
                    )
                    continue
                file_receiver.receive_chunks(message)
            else:
                try:
                    req = json.loads(message)
                    data_type = req.get("type", None)

                    if data_type == "file_start":
                        if file_receiver is not None:
                            # 清理之前的文件接收器
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "Previous file transfer not completed",
                                    }
                                )
                            )
                            continue
                        file_receiver = FileReceiver(req)
                        # await ws.send(json.dumps({"type": "status", "message": "File transfer started"}))

                    elif data_type == "file_end":
                        if file_receiver is None:
                            print("file_receiver is None")
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": "No file transfer to end",
                                    }
                                )
                            )
                            continue
                        file_receiver.end()
                        # await ws.send(json.dumps({"type": "status", "message": "File transfer completed"}))

                        # 处理文件
                        try:
                            print(file_receiver.file_path)
                            loader = PyPDFLoader(file_receiver.file_path)
                            chunked_docs = []
                            text_splitter = RecursiveCharacterTextSplitter(
                                chunk_size=1000,
                                chunk_overlap=200,
                                length_function=len,
                                is_separator_regex=False,
                            )
                            async for page in loader.alazy_load():
                                chunked_doc = text_splitter.split_documents([page])
                                chunked_docs.extend(chunked_doc)
                            vector_db = FAISS.from_documents(
                                chunked_docs,
                                embedding=OllamaEmbeddings(model="nomic-embed-text"),
                                distance_strategy=DistanceStrategy.COSINE,
                            )
                            docs = vector_db.similarity_search(prompt, k=7)
                            template_messages.append(
                                HumanMessage(
                                    f"The file's name is {file_receiver.file_name}"
                                )
                            )
                            for doc in docs:
                                template_messages.append(HumanMessage(doc.page_content))

                        except Exception as e:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "error",
                                        "message": f"File processing error: {str(e)}",
                                    }
                                )
                            )
                        finally:
                            # 清理文件接收器
                            file_receiver = None

                    elif data_type == "normal_data":
                        prompt = req.get("prompt", None)
                        if prompt is None:
                            await ws.send(
                                json.dumps(
                                    {"type": "error", "message": "No prompt provided"}
                                )
                            )
                            continue
                    elif data_type == "end":
                        prompt = req.get("prompt", prompt)
                        model = ChatOllama(model="llama3.2")
                        messages = ChatPromptTemplate.from_messages(template_messages)
                        for chunk in model.stream(
                            messages.format_messages(query=prompt)
                        ):
                            await ws.send(
                                json.dumps({"type": "result", "data": chunk.content})
                            )
                        await ws.send(json.dumps({"type": "end"}))
                    else:
                        await ws.send(
                            json.dumps(
                                {"type": "error", "message": "Invalid data type"}
                            )
                        )
                except json.JSONDecodeError:
                    await ws.send(
                        json.dumps({"type": "error", "message": "Invalid JSON message"})
                    )

    except Exception as e:
        await ws.send(json.dumps({"type": "error", "message": str(e)}))
    finally:
        if file_receiver is not None:
            # 清理未完成的文件传输
            try:
                os.remove(file_receiver.file_path)
            except:
                pass
        await ws.close()

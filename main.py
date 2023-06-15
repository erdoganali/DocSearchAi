import os
import openai
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import AzureOpenAI
from colorama import Fore, Style
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from typing import List

load_dotenv()

openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")

app = FastAPI()

def get_file_list():
    file_list = []
    for filename in os.listdir("upload"):
        if filename.endswith(".txt"):
            file_list.append(os.path.join("upload", filename))
    return file_list

def combine_files():
    files = get_file_list()
    documents = []
    for filename in files:
        with open(filename, "rb") as f:
            document = f.read().decode("utf-8")
            documents.append(document)
    combined_document = " ".join(documents)
    with open("saved/combined_document.txt", "w", encoding="utf-8") as f:
        f.write(combined_document)

combine_files()

def printGreen(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def printPurple(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)

@app.post("/uploadfiles/")
async def create_upload_files(files: List[UploadFile]):
    filenames = []
    for file in files:
        file_path = f"upload/{file.filename}"
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())
        filenames.append(file_path)

    response = {
        "status": "ok",
        "filenames": filenames
    }
    return JSONResponse(content=response)



@app.post("/qa")
async def qa_endpoint(query: str):
    query = query.strip()
    if not query:
        return JSONResponse(content={
            "status": "error",
            "message": "Sorgu boş olamaz."
        })

    # Belgeleri yükleme
    document_loader = TextLoader("saved/combined_document.txt", encoding='utf-8')
    documents = document_loader.load()

    # Belge parçaları halinde ayırma ve gömme oluşturma
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    # Chroma vektör deposunu kullanarak belgeleri sıralama
    vectorstore_db = Chroma.from_documents(texts, embeddings)
    retriever = vectorstore_db.as_retriever()

    azure_llm = AzureOpenAI(deployment_name="Davinci-003", model_name="text-davinci-003", temperature=0)
    qa = RetrievalQA.from_chain_type(azure_llm, chain_type="stuff", retriever=retriever)

    printGreen(f"Soru: {query}")
    answer = qa.run(query)

    return JSONResponse(content={
        "status": "ok",
        "query": query,
        "answer": answer
    })

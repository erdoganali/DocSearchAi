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

# yardımcı fonksiyonlar PrintGreen ve PrintPurple oluşturma:

def printGreen(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def printPurple(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    file_path = f"upload/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 'txt' dosyasından belgeleri yükleme
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

    # Belge parçaları halinde ayırma ve gömme oluşturma
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

    # Chroma vektör deposunu kullanarak belgeleri sıralama
    vectorstore_db = Chroma.from_documents(texts, embeddings)
    retriever = vectorstore_db.as_retriever()

    azure_llm = AzureOpenAI(deployment_name="Davinci-003", model_name="text-davinci-003", temperature=0)
    qa = RetrievalQA.from_chain_type(azure_llm, chain_type="stuff", retriever=retriever)

    response = {
        "status": "ok",
        "filename": file.filename
    }
    return JSONResponse(content=response)

@app.get("/qa")
async def qa_endpoint(query: str, file: UploadFile = File(...)):
    query = query.strip()
    if not query:
        return JSONResponse(content={
            "status": "error",
            "message": "Sorgu boş olamaz."
        })

    file_path = f"upload/{file.filename}"
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())

    # 'txt' dosyasından belgeleri yükleme
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()

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
    printPurple(f"Cevap: {answer}")

    return JSONResponse(content={
        "status": "ok",
        "query": query,
        "answer": answer
    })


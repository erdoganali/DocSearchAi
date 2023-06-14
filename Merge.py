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

load_dotenv()

openai.api_base = os.environ.get("OPENAI_API_BASE")
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.api_type = os.environ.get("OPENAI_API_TYPE")
openai.api_version = os.environ.get("OPENAI_API_VERSION")

# create a helper functions PrintGreen and PrintPurple:

def printGreen(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def printPurple(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)


# Load documents from text file
file_path = r"analiz.txt"
loader = TextLoader(file_path, encoding='utf-8')
documents = loader.load()

# Split documents into chunks and generate embeddings
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

# Index documents using Chroma vector store
vectorstore_db = Chroma.from_documents(texts, embeddings)
retriever = vectorstore_db.as_retriever()


azure_llm = AzureOpenAI(deployment_name="Davinci-003", model_name="text-davinci-003", temperature=0)
qa = RetrievalQA.from_chain_type(azure_llm, chain_type="stuff", retriever=retriever)

query = "Veri tabanı bağlantısı nasıl oluşturulur"
printGreen("Question " + query)
printPurple("Cevap: " + qa.run(query))



from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
import config
from langchain.llms import AzureOpenAI
from helper import printPurple, printGreen
from langchain.callbacks import get_openai_callback

loader = TextLoader("analiz.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1)

vectorstore_db = Chroma.from_documents(texts, embeddings)

result = vectorstore_db.as_retriever()
print(result)

azure_llm = AzureOpenAI(deployment_name="Davinci-003", model_name="text-davinci-003", temperature=0);
with get_openai_callback() as cb:
    qa = RetrievalQA.from_chain_type(azure_llm,chain_type="stuff", retriever=vectorstore_db.as_retriever())
    query = "Veri tabanı bağlantısı nasıl oluşturulur"
    printGreen("Question " + query)
    printPurple("Cevap: " + qa.run(query))
    print(cb)
    print(cb.total_tokens)
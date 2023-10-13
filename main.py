from langchain.text_splitter import TokenTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.indexes import VectorstoreIndexCreator
from loader.JSONLoader import JSONLoader
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load crawled data
loader = JSONLoader(
    file_path='data/scraped_data.json',
    embed_keys='.text[]',
    metadata_keys='.metadata[]')

data = loader.load()

# Create a vector store based on the data
index = VectorstoreIndexCreator(
    # split the documents into chunks
    text_splitter=TokenTextSplitter(chunk_size=1000, chunk_overlap=100),
    # select embeddings
    embedding=OpenAIEmbeddings(),
    # set vectorstore
    vectorstore_cls=Chroma
).from_loaders([loader])

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1)

# Query the vector store
# While loop to keep asking questions
while True:
    query = input("What is your question? ")
    result = index.query(llm=llm, question=query, chain_type="stuff")
    print(result)
    print("------\n")
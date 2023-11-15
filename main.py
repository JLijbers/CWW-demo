import os
import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Chroma

from functions import reciprocal_rank_fusion

# Load environment variables from .env file
load_dotenv()
website_url = os.environ.get('WEBSITE_URL', 'een website')
#gpt4_path = "./models/gpt4all-falcon-q4_0.gguf"

# Streamlit settings
st.set_page_config(
    page_title=f'Stel vragen aan: {website_url}',
    page_icon="⚽",)
st.title('⚽ Chat met: Korfbaldws.nl')


@st.cache_resource(ttl='1h')
# Supporting function and class
def get_retriever():
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type='mmr', k=3)
    return retriever


# Start
# Set Chroma vector DB as retriever
retriever = get_retriever()

# Set use of chat history
msgs = StreamlitChatMessageHistory(key='langchain_messages')
memory = ConversationBufferMemory(memory_key='chat_history', chat_memory=msgs, return_messages=True)

# Initialize the ChatOpenAI model
callbacks = [StreamingStdOutCallbackHandler()]
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1, streaming=True)
#llm = GPT4All(model=gpt4_path, callbacks=callbacks, verbose=True, streaming=True)

# Generate multiple search queries
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
    ("user", "Generate multiple search queries related to: {original_query}"),
    ("user", "OUTPUT (4 queries):")])
generate_queries = prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))

# Create the Q&A chain
chain = generate_queries | retriever.map() | reciprocal_rank_fusion
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# Streamlit settings
if st.sidebar.button('Berichtgeschiedenis wissen') or len(msgs.messages) == 0:
    msgs.clear()
    msgs.add_ai_message(f'Vraag me alles over: {website_url}!')

avatars = {'human': 'user', 'ai': 'assistant'}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder='Wat wil je weten over de mooiste korfbalvereniging van Nederland?'):
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        # Run the Q&A chain
        #response = qa_chain.run(user_query, callbacks=callbacks)
        response = chain.invoke({"original_query": user_query})

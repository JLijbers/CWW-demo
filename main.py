import os
import streamlit as st
from dotenv import load_dotenv

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.vectorstores import Chroma
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Load environment variables from .env file
load_dotenv()
website_url = os.environ.get('WEBSITE_URL', 'een website')

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


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ''):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Start
# Set Chroma vector DB as retriever
retriever = get_retriever()

# Set use of chat history
msgs = StreamlitChatMessageHistory(key='langchain_messages')
memory = ConversationBufferMemory(memory_key='chat_history', chat_memory=msgs, return_messages=True)

# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.1, streaming=True)

# Create the Q&A chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, memory=memory, verbose=False
)

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
        stream_handler = StreamHandler(st.empty())
        # Run the Q&A chain
        response = qa_chain.run(user_query, callbacks=[stream_handler])

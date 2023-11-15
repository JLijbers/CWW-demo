import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter
from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from functions import custom_search


# Load environment variables from .env file
load_dotenv()
website_url = os.environ.get('WEBSITE_URL', 'een website')
#gpt4_path = "./models/gpt4all-falcon-q4_0.gguf"

# Streamlit settings
st.set_page_config(
    page_title=f'Stel vragen aan: {website_url}',
    page_icon="⚽",)
st.title('⚽ Chat met: Korfbaldws.nl')


# Start
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


# Initialize the ChatOpenAI model
#llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, streaming=True, callbacks=[stream_handler])
#llm = GPT4All(model=gpt4_path, callbacks=callbacks, verbose=True, streaming=True)

# Create the Q&A chain
template = """Geef antwoord op de vraag en haal je antwoord uit de meegegeven context.

Vraag: 
{question}

Context: 
{context}

Antwoord:
"""

qa_prompt = PromptTemplate.from_template(template)
#qa_chain = qa_prompt | llm | StrOutputParser()

# Streamlit settings
if "messages" not in st.session_state or st.sidebar.button("Berichtgeschiedenis wissen"):
    st.session_state["messages"] = [{"role": "assistant", "content": f'Vraag me alles over: {website_url}!'}]

avatars = {"human": "user", "ai": "assistant"}
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input(placeholder='Wat wil je weten over de mooiste korfbalvereniging van Nederland?'):
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    retrieved_context = custom_search(user_query, os.environ.get("GOOGLE_API_KEY"), os.environ.get("CX"))

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, streaming=True, callbacks=[stream_handler])
        qa_chain = qa_prompt | llm | StrOutputParser()
        response = qa_chain.invoke({"question": user_query, "context": retrieved_context})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)


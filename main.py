import os
import streamlit as st
from dotenv import load_dotenv
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough


from functions import custom_search, rewrite_user_query


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
# Initialize the ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, streaming=True)
#llm = GPT4All(model=gpt4_path, streaming=True)

# Create the Q&A chain
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Geef antwoord op de vraag door het antwoord te zoeken in de meegegeven context. Als het antwoord "
                   "niet in de context te vinden is geef je aan dat je het niet weet. Dit is de context: {context}."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)
memory = ConversationBufferMemory(return_messages=True, memory_key="history")
qa_chain = RunnablePassthrough.assign(history=RunnableLambda(memory.load_memory_variables) |
                                              itemgetter("history")) | prompt | llm | StrOutputParser()

# Streamlit settings
if "messages" not in st.session_state or st.sidebar.button("Berichtgeschiedenis wissen"):
    st.session_state["messages"] = [{"role": "assistant", "content": f'Vraag me alles over: {website_url}!'}]

avatars = {"human": "user", "ai": "assistant"}
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_query := st.chat_input(placeholder='Wat wil je weten over de mooiste korfbalvereniging van Nederland?'):
    # Rewrite user query to include chat history
    if len(memory.buffer) > 1:
        improved_user_query = rewrite_user_query(user_query, memory.buffer_as_str)
    else:
        improved_user_query = user_query

    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)
    retrieved_context, search_results = custom_search(improved_user_query,
                                                      os.environ.get("GOOGLE_API_KEY"),
                                                      os.environ.get("CX"))

    with st.chat_message("assistant"):
        response = qa_chain.invoke({"context": retrieved_context, "question": improved_user_query})
        memory.save_context({"question": improved_user_query}, {"output": response})
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
        with st.expander('Meer informatie'):
            # Write out the first
            if search_results != "":
                st.write(search_results[0]['link'])


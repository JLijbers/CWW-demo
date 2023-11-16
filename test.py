import os
from operator import itemgetter
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

from functions import custom_search, rewrite_user_query


# Load environment variables from .env file
load_dotenv()
#gpt4_path = "./models/gpt4all-falcon-q4_0.gguf"

# Start
# Initialize the LLM-model
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

# Run
user_query = "Wie is de voorzitter?"
retrieved_context = custom_search(user_query, os.environ.get("GOOGLE_API_KEY"), os.environ.get("CX"))
response = qa_chain.invoke({"context": retrieved_context, "question": user_query})
memory.save_context({"question": user_query}, {"output": response})
print(response)

user_query_2 = "Wie is de wedstrijdsecretaris voor de senioren?"
test = rewrite_user_query(user_query_2, memory.buffer_as_str)

retrieved_context_2 = custom_search(user_query, os.environ.get("GOOGLE_API_KEY"), os.environ.get("CX"))
response_2 = qa_chain.invoke({"context": retrieved_context_2, "question": user_query_2})
memory.save_context({"question": user_query_2}, {"output": response_2})
print(response_2)

user_query_3 = "En wie voor de jeugd?"

if len(memory.buffer) > 0:
    improved_user_query = rewrite_user_query(user_query_3, memory.buffer_as_str)
else:
    improved_user_query = user_query_3

print(improved_user_query)

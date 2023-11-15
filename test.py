import os
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

from functions import custom_search

# Load environment variables from .env file
user_query = "Wie is de voorzitter?"
load_dotenv()
#gpt4_path = "./models/gpt4all-falcon-q4_0.gguf"

# Start
# Initialize the LLM-model
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1, streaming=True)
#llm = GPT4All(model=gpt4_path, callbacks=callbacks, verbose=True, streaming=True)

# Get context by performing a custom google search
retrieved_context = custom_search(user_query, os.environ.get("GOOGLE_API_KEY"), os.environ.get("CX"))

# Create the Q&A chain
template = """Geef antwoord op de vraag en haal je antwoord uit de meegegeven context.

Vraag: 
{question}

Context: 
{context}

Antwoord:
"""

qa_prompt = PromptTemplate.from_template(template)
qa_chain = qa_prompt | llm | StrOutputParser()

response = qa_chain.invoke({"question": user_query, "context": retrieved_context})
#print(response)

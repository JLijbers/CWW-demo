from bs4 import BeautifulSoup
import requests
import json

from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate


def find_paragraph(full_text, snippet):
    snippet_parts = [part.strip() for part in snippet.split('...') if len(part.strip()) > 1]
    sentences = full_text.split('.')

    paragraphs = []
    for part in snippet_parts:
        snippet_start = full_text.find(part)
        if snippet_start == -1:
            continue

        # Find the index of the sentence that contains the snippet
        snippet_sentence_index = -1
        current_length = 0
        for i, sentence in enumerate(sentences):
            current_length += len(sentence) + 1
            if current_length > snippet_start:
                snippet_sentence_index = i
                break

        if snippet_sentence_index == -1:
            continue

        # Select 5 sentences before and after the snippet sentence
        start_index = max(0, snippet_sentence_index - 5)
        end_index = min(len(sentences), snippet_sentence_index + 6)  # +6 to include the snippet sentence and 5 after

        # Reconstruct the paragraph
        paragraph = '.'.join(sentences[start_index:end_index]) + '.'
        paragraphs.append(paragraph.strip())

    return '\n\n'.join(paragraphs)


def custom_search(query, api_key, cx):
    url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}"
    response = requests.get(url)
    data = json.loads(response.text)

    # Initialize variables
    context = ""
    search_results = ""

    # Check for errors or empty search results
    if 'error' in data:
        return context, search_results
    elif 'items' not in data:
        return context, search_results
    else:
        # Extract search results
        search_results = data['items']

        for result in search_results:
            page_url = result['link']
            snippet = result['snippet']

            # Get full page content from page_url
            page_content = requests.get(page_url).text
            soup = BeautifulSoup(page_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)

            # Filter specific snippet-paragraphs from full text
            paragraphs = find_paragraph(text, snippet)
            context += f"Zoekresultaat-snippet: {snippet}\nBijbehorende paragraaf: {paragraphs}\n\n"

    return context, search_results


def rewrite_user_query(user_query: str, memory: str):
    # Use few-shot prompting
    template = """Hier zijn enkele voorbeelden van hoe je een gesprek tussen een Human en een AI kunt verduidelijken. 
Gebruik deze voorbeelden als gids om vergelijkbare taken uit te voeren:

Voorbeeld 1:
Gespreksgeschiedenis:
'Human: Wat is het hoofdingrediënt in een Caesar salade?
AI: Het hoofdingrediënt in een Caesar salade is Romeinse sla.

Nieuwe vraag van de Human:
'En voor een Griekse salade?''

Uitgevoerde taak:
Wat is het hoofdingrediënt in een Griekse salade?

Voorbeeld 2:
Gespreksgeschiedenis:
'Human: Hoe laat begint de film?
AI: De film begint om 20:00 uur.
Human: Hoe laat begint het concert?'
AI: Het concert begint om 21:00 uur.

Nieuwe vraag van de Human:
'En waar is die precies?'

Uitgevoerde taak:
En waar is het concert precies?

Voorbeeld 3:
Gespreksgeschiedenis:
'Human: Wie heeft de Mona Lisa geschilderd?
AI: De Mona Lisa is geschilderd door Leonardo da Vinci.
Human: En wie schilderde de Sterrennacht?'
AI: De Sterrennacht is geschilderd door Vincent van Gogh.

Nieuwe vraag van de Human:
'Zijn die schilderijen in dezelfde periode geschilderd?

Uitgevoerde taak:
'Zijn de Mona Lisa en de Sterrennacht in dezelfde periode geschilderd?

Jouw taak is om deze methode te gebruiken om vergelijkbare vragen in gesprekken te verduidelijken. 
Zorg ervoor dat je de nieuwe vraag van de Human combineert met relevante informatie uit de gespreksgeschiedenis om een 
duidelijke en specifieke vraag te vormen voor de AI. Gebruik dezelfde bewoording. 
Voer deze taak uit zonder vervolgvragen te stellen. Let's think step-by-step.

Gespreksgeschiedenis:
{memory}

Nieuwe vraag van de Human:
{user_query}

Uitgevoerde taak:
"""
    qa_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
    qa_chain = qa_prompt | llm | StrOutputParser()
    new_user_query = qa_chain.invoke({"memory": memory, "user_query": user_query})

    return new_user_query


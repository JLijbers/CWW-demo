from bs4 import BeautifulSoup
import requests
import json


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
    context = ""

    # Check for errors or empty search results
    if 'error' in data:
        print("Error:", data['error']['message'])
    elif 'items' not in data:
        print("No search results found.")
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

    return context


from langchain.docstore.document import Document
from langchain.utilities import ApifyWrapper
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

apify = ApifyWrapper()

# Call the Actor to obtain text from the crawled webpages
run = apify.call_actor(
    actor_id="apify/website-content-crawler",
    run_input={"startUrls": [{"url": "https://www.korfbaldws.nl/"}]},
    dataset_mapping_function=lambda item: Document(
        page_content=item["text"] or "", metadata={"source": item["url"]}
    ),
)


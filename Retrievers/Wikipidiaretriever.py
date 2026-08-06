from langchain_community.retrievers import WikipediaRetriever
from dotenv import load_dotenv
import wikipedia

# Load environment variables from .env (not strictly needed here since this
# script doesn't call OpenAI, but harmless to keep if you reuse this file)
load_dotenv()

# Set a custom User-Agent to avoid Wikimedia's rate-limiting of the default
# User-Agent used by the underlying `wikipedia` package (see the
# JSONDecodeError issue from earlier)
wikipedia.set_user_agent("my-langchain-app/0.1 (myemail@example.com)")

# Initialize the retriever
#   - top_k_results: how many Wikipedia articles to fetch and return
#   - lang: which language edition of Wikipedia to search (e.g. "en", "hi")
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# The natural-language query used to search Wikipedia and rank results
query = "the geopolitical history of india and pakistan from the perspective of a chinese"

# Run the search and retrieve matching Wikipedia pages as LangChain Documents
# (each Document's page_content holds the article text/summary, and
# metadata holds fields like title and source URL)
docs = retriever.invoke(query)

#print(docs)

for i, doc in enumerate(docs):
    print(f"\n--- Result {i+1} ---")
    print(f"Content:\n{doc.page_content}...")

    
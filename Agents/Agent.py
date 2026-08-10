import os
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

search_tool = DuckDuckGoSearchRun()


@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    """
    api_key = os.environ.get("WEATHERSTACK_API_KEY")
    url = f"https://api.weatherstack.com/current?access_key={api_key}&query={city}"

    response = requests.get(url)
    data = response.json()
    print("WEATHERSTACK RAW RESPONSE:", data)  # temporary debug line
    return data


llm = ChatOpenAI()

# Modern LangChain (1.x) agent — runs on LangGraph under the hood.
# No hub.pull("hwchase17/react") needed: tool-calling models don't use the
# text-based ReAct prompt format, just a plain system prompt string.
agent = create_agent(
    model=llm,
    tools=[search_tool, get_weather_data],
    system_prompt="You are a helpful assistant. Use the tools available to answer the user's question.",
)

response = agent.invoke(
    {"messages": [{"role": "user", "content": "Find the capital of Madhya Pradesh, then find it's current weather condition"}]}
)

# The final answer is the last message in the returned message list
print(response["messages"][-1].content)
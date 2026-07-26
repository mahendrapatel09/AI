from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.3, max_completion_tokens=1000)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
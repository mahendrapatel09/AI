"""
Tool calling and agent examples with LangChain.

Demonstrates:
  1. Manual tool binding + tool-call execution (single tool: multiply)
  2. Manual multi-tool chaining (currency conversion, two dependent tools)
  3. Agent-based execution with initialize_agent (classic ReAct-style agent)
"""

import json
import os
from typing import Annotated

import requests
from dotenv import load_dotenv
from langchain_classic.agents import AgentType, initialize_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from langchain_openai import ChatOpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Add it to your .env file, e.g.\n"
        "OPENAI_API_KEY=your_key_here"
    )

EXCHANGE_RATE_API_KEY = os.getenv("EXCHANGE_RATE_API_KEY")
if not EXCHANGE_RATE_API_KEY:
    raise RuntimeError(
        "EXCHANGE_RATE_API_KEY is not set. Add it to your .env file, e.g.\n"
        "EXCHANGE_RATE_API_KEY=your_key_here"
    )

# ChatOpenAI() picks up OPENAI_API_KEY from the environment automatically;
# the check above just fails fast with a clear message if it's missing.
llm = ChatOpenAI()


# ---------------------------------------------------------------------------
# 1. Single tool: multiply
# ---------------------------------------------------------------------------

@tool
def multiply(a: int, b: int) -> int:
    """Given two numbers a and b, returns their product."""
    return a * b


def run_multiply_demo() -> str:
    """Bind a single tool to the LLM and manually execute the requested call."""
    llm_with_tools = llm.bind_tools([multiply])

    messages = [HumanMessage("can you multiply 3 with 1000")]
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    if not ai_message.tool_calls:
        return ai_message.content

    tool_result = multiply.invoke(ai_message.tool_calls[0])
    messages.append(tool_result)

    return llm_with_tools.invoke(messages).content


# ---------------------------------------------------------------------------
# 2. Two dependent tools: currency conversion
# ---------------------------------------------------------------------------

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """Fetch the currency conversion factor between a base and target currency."""
    url = (
        f"https://v6.exchangerate-api.com/v6/{EXCHANGE_RATE_API_KEY}"
        f"/pair/{base_currency}/{target_currency}"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg],
) -> float:
    """Convert a base currency value into the target currency using a given rate."""
    return base_currency_value * conversion_rate


def run_currency_conversion_demo() -> str:
    """Bind two dependent tools, execute the first, feed its result into the second."""
    llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

    messages = [
        HumanMessage(
            "What is the conversion factor between INR and USD, and based on "
            "that can you convert 10 inr to usd"
        )
    ]
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    conversion_rate = None
    for tool_call in ai_message.tool_calls:
        if tool_call["name"] == "get_conversion_factor":
            tool_message = get_conversion_factor.invoke(tool_call)
            conversion_rate = json.loads(tool_message.content)["conversion_rate"]
            messages.append(tool_message)

        elif tool_call["name"] == "convert":
            if conversion_rate is None:
                raise RuntimeError(
                    "convert was called before get_conversion_factor resolved a rate"
                )
            tool_call["args"]["conversion_rate"] = conversion_rate
            tool_message = convert.invoke(tool_call)
            messages.append(tool_message)

    return llm_with_tools.invoke(messages).content


# ---------------------------------------------------------------------------
# 3. Agent-based execution (classic ReAct-style agent)
# ---------------------------------------------------------------------------

def run_agent(user_query: str) -> str:
    """Run the currency tools through a classic initialize_agent ReAct agent.

    Note: initialize_agent/AgentType is deprecated in LangChain v1 in favor of
    `create_agent` (from langchain.agents). Kept here since it still works via
    langchain_classic — migrate when convenient.
    """
    agent_executor = initialize_agent(
        tools=[get_conversion_factor, convert],
        llm=llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )
    return agent_executor.invoke({"input": user_query})["output"]


if __name__ == "__main__":
    print("--- Multiply demo ---")
    print(run_multiply_demo())

    print("\n--- Currency conversion demo ---")
    print(run_currency_conversion_demo())

    print("\n--- Agent demo ---")
    print(run_agent("Hi how are you?"))
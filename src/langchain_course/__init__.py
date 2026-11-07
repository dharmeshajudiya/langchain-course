from dotenv import load_dotenv

load_dotenv()

from typing import List

from langchain.agents import create_agent  # to create agent
from langchain.tools import tool  # agent need tools to perform actions
from langchain_core.messages import (
    HumanMessage,
)  # agent need messages to communicate with user
from langchain_openai import ChatOpenAI  # agent need llm for reasoning

# from tavily import TavilyClient
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent."""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent's response with answer and sources."""

    answer: str = Field(description="The  agent's answer to the user's query")
    sources: List[Source] = Field(
        default_factory=list, description="List of sources used to generate the answer"
    )


tavily_search = TavilySearch(max_results=3, topic="general")

llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm_model,
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information.",
    response_format=AgentResponse
)


def main() -> None:
    print("Creating agent...")
    message = HumanMessage(
        content="I want the 3 job posting in LangChain in Vadodara, Rajkot and Ahmedabad and list here."
    )
    result = agent.invoke({"messages": [message]})
    print(f"Agent response: {result}")


if __name__ == "__main__":
    main()
 
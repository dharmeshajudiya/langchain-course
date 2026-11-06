from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent  # to create agent
from langchain.tools import tool  # agent need tools to perform actions
from langchain_core.messages import HumanMessage  # agent need messages to communicate with user
from langchain_openai import ChatOpenAI  # agent need llm for reasoning
#from tavily import TavilyClient
from langchain_tavily import TavilySearch

tavily_search = TavilySearch(max_results=3, topic="general")

llm_model = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_agent(
    model=llm_model, 
    tools=[tavily_search],
    system_prompt="You are a helpful research assistant. Use web search to find accurate, up-to-date information."
)

def main() -> None:
    print("Creating agent...")
    message = HumanMessage(content="I want the 3 job posting in LangChain in Vadodara, Rajkot and Ahmedabad and list here.")
    result = agent.invoke({"messages": [message]})
    print(f"Agent response: {result}")

if __name__ == "__main__":
    main()
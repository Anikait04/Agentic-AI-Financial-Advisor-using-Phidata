from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

openai_key=os.getenv("OPENAI_API_KEY")

load_dotenv()

#websearch agent
websearch_agent=Agent(
    name='websearch_agent',
    role="search the web for the information",
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[DuckDuckGo()],
    instructions=["always include sources of information"],
    show_tool_calls=True,
    markdown=True,
)

## financial agent
financial_agent=Agent(
    name='financial_agent',
    model=Groq(id="llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True, stock_fundamentals=True,
                         company_news=True)],
    instructions=["use tables to display the data"],
    show_tool_calls=True,
    markdown=True,
    
)
multi_ai_agents=Agent(
    team=[websearch_agent,financial_agent],
    model=Groq(id="llama-3.1-70b-versatile"),
    instructions=["Always include sources of information","use tables to display the data"],
    show_tool_calls=True,
    markdown=True
    )
multi_ai_agents.print_response("Summarize analyst recommendations and share the latest news for Apple",stream=True)

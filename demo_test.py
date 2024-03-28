import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY", None)

from json import tool
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# if api_key:
#   print(f"OpenAI API Key: {api_key}")
# else:
#   print("Please set the OPENAI_API_KEY environment variable.")

# Topic for the crew run
crewTopic = "Latest AI Trends and Insights"

# Creating a researcher agent with memory and verbose mode
agent_researcher = Agent(
    role="Researcher",
    goal=f"Research {crewTopic}",
    verbose=True,
    memory=True,
    backstory=f"""You are an AI research assistant who specializes in writing about {
        crewTopic}.""",
    allow_delegation=False,
)

# Creating a writer agent with custom tools and delegation capability
agent_writer = Agent(
    role="Writer",
    goal=f"Write compelling and engaging blog posts about {crewTopic}",
    verbose=True,
    memory=True,
    backstory=f"""You are an AI blog post writer who specialized in writing about {
        crewTopic}.""",
    allow_delegation=False,
)

# Research task
task_research = Task(
    description=f"""Identify the next big trend in {crewTopic}.
  Focus on identifying pros and cons and the overall narrative.
  Your final report should clearly articulate the key points,
  its market opportunities, and potential risks.""",
    expected_output=f"""A comprehensive 3 paragraphs long report on the {crewTopic}.""",
    tools=[search_tool],
    agent=agent_researcher,
)

# Writing task with language model configuration
task_write = Task(
    description=f""""Compose an insightful article on {crewTopic}.
  Focus on the latest trends and how it's impacting the industry.
  This article should be easy to understand, engaging, and positive.""",
    expected_output=f"A 4 paragraph article on {crewTopic} advancements fromated as markdown.",
    tools=[search_tool],
    agent=agent_writer,
    output_file="new-blog-post.md",  # Example of output customization
)

# Forming the tech-focused crew with enhanced configurations
aiCrew = Crew(
    agents=[agent_researcher, agent_writer],
    tasks=[task_research, task_write],
    process=Process.sequential,  # Optional: Sequential task execution is default
    # verbose=2
)

# Starting the task execution process with enhanced feedback
result = aiCrew.kickoff()
print(result)

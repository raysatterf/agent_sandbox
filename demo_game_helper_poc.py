from langchain.agents import tool as lcagents_tool  # lc for LangChain
from langchain.agents import load_tools
from langchain_community.agent_toolkits import FileManagementToolkit
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool as crewai_tool
from json import tool as json_tool
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from playsound import playsound
import sys

# from langchain.llms import Ollama
# from langchain.tools import DuckDuckGoSearchRun

import os

client_openai = OpenAI()

working_dir = "C:/dev/proj/demo_crewai-test-2/working_dir"
audio_output_filename = "output_audio.mp3"
audio_output_fullpath = f"{working_dir}/{audio_output_filename}"

# source_text = "Beep boop, I am en ro-bit. This is a test."

load_dotenv()  # Load environment variables from .env file


@crewai_tool(
    "DuckDuckGoSearch"
)
def websearch_duckduckgo(search_query: str):
    """Search the web for information"""
    return DuckDuckGoSearchRun().run(search_query)

api_key = os.environ.get("OPENAI_API_KEY", None)

file_management_toolkit = FileManagementToolkit()

# if api_key:
#   print(f"OpenAI API Key: {api_key}")
# else:
# print("Please set the OPENAI_API_KEY environment variable.")

# Define project directory (replace with your actual project location)
project_dir = "C:/dev/proj/demo_crewai-test-2/working_dir"

# Topic for the crew run
crewTopic = f"""Latest recommendations for the Paladin class in the 
game "Last Epoch", ensuring the recommendations are based on the state 
of the game as-of the very latest update patch for the game."""

response_voice = client_openai.audio.speech.create(
    input=crewTopic,
    model="tts-1",  # The latest text to speech model, optimized for speed.
    voice="nova",
)

# response_voice.stream_to_file(audio_output_fullpath)
# playsound(audio_output_fullpath)
# os.remove(audio_output_fullpath)

# Creating a researcher agent with memory and verbose mode
agent_researcher = Agent(
    role="Researcher",
    goal=f"Research {crewTopic}",
    verbose=True,
    memory=True,
    backstory=f"""You are a video-game research assistant who strives to be 
    the best in their field, who specializes in finding, collecting, and 
    researching information on the topic of {crewTopic}.""",
    allow_delegation=False,
)

# Creating a writer agent with custom tools and delegation capability
agent_writer = Agent(
    role="Writer",
    goal=f"Write compelling and engaging blog posts about {crewTopic}",
    verbose=True,
    memory=True,
    backstory=f"""You are a video-game content creator who strives to be the 
    best in their field, who passionately and perfectly studies the research 
    they have access to, and specializes in writing about {crewTopic}.""",
    allow_delegation=False,
)

# Research task
task_research = Task(
    description=f"""Identify the most relevant and important information on 
    the next big builds, metas, or other emerging resources, facts, gameplay 
    and more on the topic of {crewTopic}. Your final report should be clear, 
    articulate, extremely detailed, and insightful.""",
    expected_output=f"""A comprehensive 5 paragraphs long report on the {
        crewTopic}.""",
    tools=[websearch_duckduckgo],
    agent=agent_researcher,
)

# Writing task with language model configuration
task_write = Task(
    description=f""""Compose an insightful article on {crewTopic}.
  Focus on the latest news and information within the last couple weeks.
  This article should be easy to understand, engaging, and positive.""",
    expected_output=f"""A 4 paragraph article on {
        crewTopic} advancements saved as a markdown file.""",
    tools=[websearch_duckduckgo],
    agent=agent_writer,
    output_file=f"""{project_dir}/output_report.md""",  # Example of output customization
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

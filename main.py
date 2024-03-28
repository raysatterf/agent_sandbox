from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY", None)

product_name = "Oath to Embers Mobile Companion App"
product_description = """A user-friendly mobile app designed to be used (optionally) with the 'Oath to Embers'
role-playing board game as a means to enhance the gameplay experience."""

# Function to generate detailed backstories for agents

def create_backstory(role, expertise):
    """
    Creates a detailed backstory for a CrewAI agent.

    Args:
        role (str): The agent's role (e.g., Software Engineer, Product Manager).
        expertise (str): The agent's area of expertise (e.g., React, DevOps).

    Returns:
        str: A detailed backstory for the agent.
    """
    return f"""
    You are a highly skilled and detail-oriented {role} with extensive experience in modern software development practices.
    You possess in-depth knowledge of {expertise} and a passion for building high-quality, maintainable code.
    Your expertise includes best practices in logging, error handling, debugging, unit testing, and adherence to industry standards.
    You are a strong collaborator and communicator, fostering a positive and productive team environment.
    """


# Define project directory (replace with your actual project location)
project_dir = "C:/dev/proj/demo_crewai-test-2/working_dir"

# Define tool instances
search_tool = DuckDuckGoSearchRun()
file_management_toolkit = FileManagementToolkit()

# Define Crew roles and their corresponding expertise
roles = {
    "Product Manager": "User research, market analysis, product requirements",
    "Business Analyst": "Business process analysis, user stories, system requirements",
    "Software Engineer (Frontend)": "React, JavaScript, HTML, CSS",
    "Software Engineer (Backend)": "Python, Django, Flask, APIs",
    "Senior Engineer": "Full-stack development, architecture design, mentorship, code review",
    "QA Tester": "Test automation, manual testing, test case design",
    "UAT Tester": "Acceptance testing, user experience testing",
    "Security Engineer": "Code security audits, vulnerability assessments",
    "DevOps Engineer": "Infrastructure management, deployment automation",
    "UI/UX Designer": "User interface design, user experience design",
}

# Create CrewAI agents
agents = {}
for role, expertise in roles.items():
    agent = Agent(
        role=role,
        goal=f"Fulfill the responsibilities of a {role} effectively.",
        verbose=True,
        memory=True,
        backstory=create_backstory(role, expertise),
        allow_delegation=True  # Allow agents to delegate tasks if needed
    )
    agents[role] = agent

# Define helper function to create tasks with common properties
def create_task(description, expected_output, agent, tools=[]):
    return Task(
        description=description,
        expected_output=expected_output,
        agent=agent,
        tools=tools
    )

# Product Manager tasks
product_mgr = agents["Product Manager"]
tasks = [
    create_task(
        description="Conduct user research to identify user needs and pain points.",
        expected_output="""A comprehensive report on user research findings. This includes competitor products such as
        the 'Oathsworn' app (https://www.shadowborne-games.com/pages/oathapp), a detailed feature list
        of competitor products (if available), and other supporting information and details that
        would serve as a basis for a baseline minimum viable product and/or goal.""",
        agent=product_mgr,
        tools=[search_tool]
    ),
    create_task(
        description="Develop a product roadmap outlining key features and functionalities.",
        expected_output="A prioritized product roadmap with timelines.",
        agent=product_mgr,
        tools=[file_management_toolkit]  # Use for creating roadmap document,
    ),
]

# Business Analyst tasks
business_analyst = agents["Business Analyst"]
tasks.extend([
    create_task(
        description="Analyze user stories and translate them into detailed system requirements.",
        expected_output="A document outlining technical requirements for development.",
        agent=business_analyst,
        tools=[search_tool]
    ),
    create_task(
        description="Create user story maps to visualize user journeys and interactions.",
        expected_output="A user story map depicting user flows and functionalities.",
        agent=business_analyst
    ),
])

# Software Engineer tasks
frontend_engineer = agents["Software Engineer (Frontend)"]
backend_engineer = agents["Software Engineer (Backend)"]
senior_engineer = agents["Senior Engineer"]
tasks.extend([
    create_task(
        description="Develop the user interface (UI) components for the application using React.",
        expected_output="Functional and responsive UI components for the application.",
        agent=frontend_engineer,
        tools=[search_tool, file_management_toolkit]  # Use for code & file management
    ),
    create_task(
        description="Write clean, maintainable, and well-tested code using modern JavaScript practices.",
        expected_output="Unit tests covering core functionalities of developed UI components.",
        agent=frontend_engineer,
        tools=[file_management_toolkit]  # Use for code storage
    ),
    create_task(
        description="Develop the backend API for the application using Python and a framework like Django or Flask.",
        expected_output="A functional and secure backend API with documented endpoints.",
        agent=backend_engineer,
        tools=[search_tool, file_management_toolkit]  # Use for code & file management
    ),
    create_task(
        description="Implement unit tests for the backend API to ensure code quality and stability.",
        expected_output="Unit tests covering core functionalities of the developed backend API.",
        agent=backend_engineer,
        tools=[file_management_toolkit]  # Use for code storage
    ),
    create_task(
        description="Guide and oversee development efforts.",
        expected_output="Successful completion of development tasks by backend and frontend engineers.",
        agent=senior_engineer,
        tools=[
            search_tool,  # For general research
            "pylint",  # Linter for code review
            "mypy",  # Static type checker for Python code
            "slack"  # Communication platform (replace with your actual tool)
        ]
    )
])

# Security Engineer tasks
security_engineer = agents["Security Engineer"]
tasks.extend([
    create_task(
        description="Perform code security audits to identify vulnerabilities in the application.",
        expected_output="A report outlining security vulnerabilities and recommendations for remediation.",
        agent=security_engineer,
        tools=[search_tool, file_management_toolkit]  # Use for code access
    )
])

# DevOps Engineer tasks
devops_engineer = agents["DevOps Engineer"]
tasks.extend([
    create_task(
        description="Set up and configure a CI/CD pipeline for automated deployment.",
        expected_output="A functional CI/CD pipeline for the project.",
        agent=devops_engineer,
        tools=[search_tool]
    )
])

# UI/UX Designer tasks
ui_ux_designer = agents["UI/UX Designer"]
tasks.extend([
    create_task(
        description="Design user interfaces (UI) that are user-friendly and aesthetically pleasing.",
        expected_output="Mockups and prototypes for the application's user interface.",
        agent=ui_ux_designer,
        tools=[search_tool]  # Use for design research
    )
])

# QA Tester tasks
qa_tester = agents["QA Tester"]
tasks.extend([
    create_task(
        description="Develop automated test scripts to ensure application functionality.",
        expected_output="Automated test scripts covering key user flows and functionalities.",
        agent=qa_tester,
        tools=[search_tool, file_management_toolkit]  # Use for test script storage
    ),
    create_task(
        description="Perform exploratory and regression testing to identify and report bugs.",
        expected_output="Detailed bug reports with steps to reproduce and potential fixes.",
        agent=qa_tester
    )
])

# UAT Tester tasks
uat_tester = agents["UAT Tester"]
tasks.extend([
    create_task(
        description="Evaluate the application from a user perspective and identify usability issues.",
        expected_output="Usability testing report with recommendations for improvement.",
        agent=uat_tester,
        tools=[search_tool]
    ),
    create_task(
        description="Perform acceptance testing to ensure the application meets user requirements.",
        expected_output="Pass/Fail report on acceptance criteria for the application.",
        agent=uat_tester
    )
])

# Team Lead tasks
# Team Lead can also be the Project Manager
team_lead = agents["Project Manager"]
tasks.extend([
    create_task(
        description="Manage the project timeline and track team progress according to Agile methodologies.",
        expected_output="Updated project status reports and backlog updates.",
        agent=team_lead,
        tools=[file_management_toolkit]  # Use for document storage
    ),
    create_task(
        description="Facilitate communication and collaboration between team members.",
        expected_output="Minutes from team meetings and clear action items.",
        agent=team_lead
    )
])

# Main Crew (Hierarchical) with all roles and sub-crews
main_crew = Crew(
    agents=[product_mgr, business_analyst, security_engineer,
            devops_engineer, ui_ux_designer, qa_tester, uat_tester,
            team_lead, frontend_engineer, backend_engineer, senior_engineer],
    tasks=tasks,
    process=Process.hierarchical
)

# Start the Crew execution with feedback
result = main_crew.kickoff()
print(result)

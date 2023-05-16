from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.agents import initialize_agent, Tool, AgentType

from tools.tools import get_profile_url


def linkedin_lookup_agent(name: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    template = """given the full name {name_of_person} I want you to 
    get me a link to their Linkedin page.
    Your answer should only contain a URL"""

    # URL will be passed to the scape_linkedin_profile function
    tools_for_agent = [
        Tool(
            name="Crawl Google 4 linkedin profile page",
            func=get_profile_url,
            description="useful for when you need to get the Linkedin profile page URL",
        )
    ]

    agent = initialize_agent(
        tools=tools_for_agent,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        llm=llm,
    )

    prompt_template = PromptTemplate(
        template=template, input_variables=["name_of_person"]
    )

    linkedin_profile_url = agent.run(prompt_template.format_prompt(name_of_person=name))

    return linkedin_profile_url

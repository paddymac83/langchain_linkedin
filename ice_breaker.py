from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from agents.twitter_lookup_agent import lookup as twitter_lookup_agent
from third_parties.twitter import scrape_user_tweets

name = "Harrison Chase"

if __name__ == "__main__":
    print("Hello Langchain")

    linkedin_profile_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)

    twitter_username = twitter_lookup_agent(name=name)
    # tweets = scrape_user_tweets(username="?", num_tweets=100)

    summary_template = """
    given the information {information} about a person I want you to create
    1. a short summary of the person
    2. two interesting facts about the person
    3. a topic that you think the person is interested in
    4. a creative icebreaker to startup a conversation with the person
    """

    # create prompy template
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # create chat model, temp=0 means its not creative
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    print(chain.run(information=linkedin_data))

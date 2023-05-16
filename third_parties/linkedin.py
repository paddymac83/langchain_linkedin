import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """Scrape information from the linkedin profile
    manually scrape information from the linkedin profile
    """

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    # create code that removes empty keys in response.json()
    data = response.json()

    data = {
        k: v
        for k, v in data.items()
        if v not in [None, "", "", [], {}]
        and k not in ["people_also_viewed", "certifications"]
    }

    # remove profile_pic_url from data, additional filtering
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data

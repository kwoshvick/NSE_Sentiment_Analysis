import tweepy
import xml.etree.ElementTree as ET

class Credentials:
    def __init__(self):
        self.credential_xml = 'twitter-credentials.xml'

    def get_twitter_credentials(self):
        credential_xml_data = ET.parse(self.credential_xml).getroot()

        return (credential_xml_data[0].text,
                credential_xml_data[1].text,
                credential_xml_data[2].text,
                credential_xml_data[3].text)


    def authentinticate_twitter(self):
        twitter_credentials = self.get_twitter_credentials()
        auth = tweepy.OAuthHandler(twitter_credentials[0], twitter_credentials[1])
        auth.set_access_token(twitter_credentials[2], twitter_credentials[3])
        api = tweepy.API(auth)
        return api



from googleapiclient.discovery import build
import os
import dotenv

dotenv.load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")


def get_youtube_service():
    return build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)



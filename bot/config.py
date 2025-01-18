import os
from dotenv import load_dotenv

load_dotenv()

class BotConfig:
    def __init__(self):
        self.port = int(os.getenv("PORT", 3978))
        self.app_id = os.getenv("MICROSOFT_APP_ID", "")
        self.app_password = os.getenv("MICROSOFT_APP_PASSWORD", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_endpoint = os.getenv("OPENAI_ENDPOINT", "")
        self.openai_deployment_name = os.getenv("OPENAI_DEPLOYMENT_NAME", "")

    def get_openai_config(self):
        return {
            "api_key": self.openai_api_key,
            "endpoint": self.openai_endpoint,
            "deployment_name": self.openai_deployment_name
        }

    def get_bot_config(self):
        return {
            "port": self.port,
            "app_id": self.app_id,
            "app_password": self.app_password
        }

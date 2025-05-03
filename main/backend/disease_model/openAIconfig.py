# azure_openai_config.py  
import os  
  
class AzureOpenAIConfig:  
    def __init__(self):  
        self.endpoint = os.getenv("ENDPOINT_URL", "https://openai-api-pcd-sym.openai.azure.com/")  
        self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")  
        self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")  
      
    def get_endpoint(self):  
        return self.endpoint  
  
    def get_deployment(self):  
        return self.deployment  
  
    def get_subscription_key(self):  
        return self.subscription_key  
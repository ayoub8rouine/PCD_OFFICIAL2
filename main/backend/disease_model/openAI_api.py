import os
import base64
from openai import AzureOpenAI
from configopenAI import AzureOpenAIConfig  # Make sure the file is named 'openAIconfig.py'

#AzurOpenAIConfig definition create a file that have this class named configopenAI.py and change the parametre
# class AzureOpenAIConfig:  
#     def __init__(self):  
#         self.endpoint = os.getenv("ENDPOINT_URL", "https://openai-api-pcd-sym.openai.azure.com/")  
#         self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")  
#         self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "REPLACE_WITH_YOUR_KEY_VALUE_HERE")  
      
#     def get_endpoint(self):  
#         return self.endpoint  
  
#     def get_deployment(self):  
#         return self.deployment  
  
#     def get_subscription_key(self):  
#         return self.subscription_key  

# Initialize config
config = AzureOpenAIConfig()

# Retrieve config values from the class
endpoint = config.get_endpoint()
deployment = config.get_deployment()
subscription_key = config.get_subscription_key()

# Initialize Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

# Optional: encode image
# IMAGE_PATH = "YOUR_IMAGE_PATH"
# encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')

# Prepare the chat prompt
chat_prompt = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are an AI assistant that helps people find information."
            }
        ]
    }
]

# Include speech result if speech is enabled
messages = chat_prompt

# Generate the completion
completion = client.chat.completions.create(
    model=deployment,
    messages=messages,
    max_tokens=800,
    temperature=0.7,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None,
    stream=False
)

print(completion.to_json())

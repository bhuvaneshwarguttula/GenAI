from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7,
)

result = model.invoke("What is the capital of india")

'''
content='The capital of India is **New Delhi**.' additional_kwargs={} 
response_metadata={'finish_reason': 'STOP', 'model_name': 'gemini-2.5-flash', 
'safety_ratings': [], 'model_provider': 'google_genai'} 
id='lc_run--09-0' tool_calls=[] 
invalid_tool_calls=[] usage_metadata={'input_tokens': 7, 'output_tokens': 24, 
'total_tokens': 31, 'input_token_details': {'cache_read': 0}, 
'output_token_details': {'reasoning': 15}}
'''

print(result.content)


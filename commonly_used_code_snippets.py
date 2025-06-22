#11/08/24


#--------------Setting Up env and keys using dotenv-------------------------
import os
from dotenv import load_dotenv
# Specify the path to your central .env file
dotenv_path = 'S:/Projects/Personal/_env/.env' # .env is the file name 
load_dotenv(dotenv_path)

# Accessing the API key
api_key = os.getenv('OPENAI_API_KEY')

# Use your API key
print(api_key)
#---------------------------------------------------



#------------------LM Studio---------------------
import os
from langchain_openai import ChatOpenAI
os.environ["OPENAI_API_KEY"] = "NA"
lm_llm = ChatOpenAI(
    model = "TheBloke/Meta-Llama-3-8B-Instruct-Q4_K_M-gguf",
    base_url ="http://localhost:1234/v1",
    api_key=""
)
#---------------------------------------------------


#-------------------ollama--------------------
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
response = llm.invoke("Name a fruit")
print (response)
#---------------------------------------------------




#AI Helpers
# Claude
# OpenAI o1/o3
# gemini/aistudio
# microsoft copilot
# cursor
# github copilot
# gemini code assisst
# grok 
# deep seek via groq
# deep seel local via ollama
# quen cloud and local
# codeium


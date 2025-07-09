from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192",
    temperature=0.0,
)



result = llm.invoke("Tell about Jharkhand")
print(result.content)

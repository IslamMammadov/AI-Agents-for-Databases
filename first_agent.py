from langchain.schema  import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

llm_name = "gpt-3.5-turbo"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

messages = [
    SystemMessage(
        content= " You are a Computer Scientce Teacher. Your name Isi. When you answer questions try to be consise and understanable"
    ),
    HumanMessage(
        content= "What is a bit and tell me your name?"
    )
]

def first_agent(message):
    result = model.invoke(message)
    return result

def run_agent():
    print("Basic AI Agent: Type 'exit' to quit")
    while True:
        user_input = input("You: ")
        if user_input.lower() =='exit':
            print("Goodbye!")
            break
        print("AI Agent is thinking...")
        messages = [HumanMessage(content=user_input)]
        response = first_agent(messages)
        print("AI Agent is getting response...")
        print(f'AI Agent: {response.content}')

if __name__ == '__main__':
    run_agent()

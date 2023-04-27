import os
import gradio as gr
from langchain.chat_models import ChatOpenAI
from langchain import LLMMathChain
from langchain.utilities import GoogleSerperAPIWrapper, WikipediaAPIWrapper
from langchain.agents import initialize_agent, Tool, AgentType


def initialize():
    # Define models
    llm = ChatOpenAI(temperature=0) 

    search = GoogleSerperAPIWrapper()
    wiki = WikipediaAPIWrapper(top_k_results = 1)
    llm_math_chain = LLMMathChain(llm=llm)
    tools = [
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        ),
        Tool( 
            name = "wikipedia",
            func=wiki.run,
            description="useful for when you need to answer questions about historical entity. the input to this should be a single search term."
        ),
        Tool(
            name = "Current Search",
            func=search.run,
            description="useful for when you need to answer questions about current events or the current state of the world, also useful if there is no wikipedia result. the input to this should be a single search term."
        )
    ]

    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chatbot_engine = initialize_agent(
        tools,
        llm, 
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
        verbose=True, 
        memory=memory)
    
    return chatbot_engine


def set_key(api_key):
    if not api_key:
        return "Key can't be empty!", None
    
    os.environ["OPENAI_API_KEY"] = api_key

    chatbot_engine = initialize()
    return "Key received", chatbot_engine


def chat(chat_history, chatbot_engine, message=""):
    # Empty msg
    if not message.strip():
        return chat_history, chat_history, ""
    
    # Execute message
    try:
        result = chatbot_engine.run(message.strip())
    except ValueError:
        result = "I can't handle this request, please try something else."

    chat_history.append((message, result))
    return chat_history, chat_history, ""


with gr.Blocks() as demo:
    # Declearing states    
    chat_history = gr.State([])
    chatbot_engine = gr.State()

    with gr.Row():
        openai_api_key_textbox = gr.Textbox(
            placeholder="Paste your OpenAI API key (sk-...)",
            show_label=False,
            lines=1,
            type="password",
        )

        api_key_set = gr.Button("Set key")

        api_key_set.click(
            fn=set_key,
            inputs=[openai_api_key_textbox],
            outputs=[api_key_set, chatbot_engine],
        )

    gr.Markdown("""<h1><center>Chat with your online-connected bot!</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    submit = gr.Button("SEND")
    submit.click(chat, inputs=[chat_history, chatbot_engine, message], outputs=[chatbot, chat_history, message])


if __name__ == "__main__":
    demo.launch(debug = True)
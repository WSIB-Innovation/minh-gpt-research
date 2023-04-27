import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_TYPE"]=os.getenv("OPENAI_API_TYPE")
os.environ["OPENAI_API_VERSION"]=os.getenv("OPENAI_API_VERSION")
os.environ["OPENAI_API_BASE"]=os.getenv("OPENAI_API_BASE")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

import gradio as gr

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import ChatVectorDBChain


# Define models
embeddings = OpenAIEmbeddings(model="my-embedding", chunk_size=1)
llm35_chat = AzureChatOpenAI(
    model_name="gpt-35-turbo",
    deployment_name="my-chatbot-35",
    max_tokens=1200
)

# Define prompts
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

system_template="""Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If the query is unrelated, just say that "The query is unrelated to the document".
The query consists of previous conversation between the user and you.
----------------
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)


def convert_PDF(pdf_obj, chatbot_engine):
    # Check if pdf_obj is a `list`
    if isinstance(pdf_obj, list):
        pdf_obj = pdf_obj[0]

    # Load pdf using Unstructured
    file_path = pdf_obj.name
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    texts = text_splitter.split_documents(data) 

    print(texts[2])

    # Create a Chroma vector store (temp)
    vectorstore = Chroma.from_documents(texts, embeddings)

    # Create a ChatVectorDBChain
    chatbot_engine = ChatVectorDBChain.from_llm(
        llm35_chat,
        vectorstore,
        qa_prompt=prompt,
        return_source_documents=True,
        top_k_docs_for_context=3)

    # Character count as text output
    output = f'There are {len(data[0].page_content)} characters in your document, which is approximately {len(data[0].page_content) // 4} tokens.'

    return output, chatbot_engine


def chat(message, chat_history, chatbot_engine):
    result = chatbot_engine({"question": message, "chat_history": chat_history})
    chat_history.append((message, result["answer"]))
    return chat_history, chat_history


with gr.Blocks() as demo:
    # Declearing states    
    chat_history = gr.State([])
    chatbot_engine = gr.State()

    # Structuring interface
    text_file = gr.File(
        label="Download Text File",
        file_count="single",
        type="file"
    )
    convert_button = gr.Button("Let your bot skim through this real quick...")
    text_output = gr.Textbox()

    convert_button.click(
        fn=convert_PDF,
        inputs=[text_file, chatbot_engine],
        outputs=[text_output, chatbot_engine],
    )

    gr.Markdown("""<h1><center>Chat with your Book!</center></h1>""")
    chatbot = gr.Chatbot()
    message = gr.Textbox()
    submit = gr.Button("SEND")
    submit.click(chat, inputs=[message, chat_history, chatbot_engine], outputs=[chatbot, chat_history])

if __name__ == "__main__":
    demo.launch(debug = True)
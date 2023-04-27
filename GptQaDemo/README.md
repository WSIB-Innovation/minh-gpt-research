### A simple chatbot that can answer your question about private pdf document
This current model uses AzureOpenAI service. To use with OpenAI, please refactor.

### User flow:
- Upload PDF
- Start chatting

### Technology explaination:
- `langchain` is a library that connects LLM like ChatGPT to multiple data sources and tools.
- Your pdf document is parsed and convert to embeddings by OpenAI `ada` model, then it is saved locally in a vectorstore database `chromadb`
- For every incoming query, it is converted into embeddings. A similarity check between this embedded query and the vectorstore is performed to find the relevant documents/paragraphs, which are then passed to the chatbot.

### Tools:
- `gradio`: Build simple chatbot interface
- `langchain`: Main library that connects the bot to data source / vectorstore
- `chromadb`: Vectorstore, there are other alternatives like `pinecone`



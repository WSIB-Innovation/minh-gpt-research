### A simple chatbot that can connect to the internet

### Live Demo Here: https://huggingface.co/spaces/mvnhat/langchain-agent-demo

### User flow:
- Paste your OpenAI key, hit "Set key"
- Start chatting

### Technology explaination:
- `langchain` is a library that connects LLM like ChatGPT to multiple online data sources and tools. Doing this way, the chatbot can self-ask to pick which tool it needs on each step of it's thought process.

### Tools:
- `gradio`: Build simple chatbot interface
- `langchain`: Main library that connects the bot to data source / vectorstore
- `wikipedia`: Do queries on Wikipedia



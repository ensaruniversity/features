## Learn End to End Next-Gen AI projects - Beginner friendly - Langchain , Pinecone - OpenAI, HuggingFace &amp; LLAMA 2 models

Master LangChain Build #16 AI Apps-OpenAI,LLAMA2,HuggingFace

https://ensarsolutions.udemy.com/course/learn-langchain-go-from-zero-to-hero-build-ai-apps

- **Anaconda Installation**
- **Open AI Key Generation**

### Build Simple Conversational Application

Let's Build Simple Conversational Application\app.py
Use Session to store conversation. 
```
st.session_state.sessionMessages.append(HumanMessage(content=question))
```

### Similar Words Finder Application
Use FASS to search similar words.
Let's build Similar Words Finder Application\app.py

### Chat Model Practical Implementation
Chat Model Practical Implementation using Python\app.py
```
chat = ChatOpenAI(temperature=.7, model='gpt-3.5-turbo')
```

### Embeddings Practical Implementation
Embeddings Practical Implementation using Python/Text Embeddings Intro.ipynb
```
embeddings = OpenAIEmbeddings()
our_Text = "Hey buddy"
text_embedding = embeddings.embed_query(our_Text)
print (f"Our embedding is {text_embedding}")
```

### Using Open Source LLM
LLM+Intro.ipynb
```
llm = HuggingFaceHub(repo_id = "google/flan-t5-large")
```

### PromptTemplate to Query

```
prompt = PromptTemplate(
    template="Provide 5 examples of {query}.\n{format_instructions}",
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions}
)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")
prompt = prompt.format(query="Currencies")
```

### FewShotPromptTemplate with Example Query selection

```
llm = OpenAI(temperature=.9, model="gpt-3.5-turbo-instruct")
new_prompt_template = FewShotPromptTemplate(
        example_selector=example_selector,  # use example_selector instead of examples
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["template_userInput","template_ageoption","template_tasktype_option"],
        example_separator="\n"
    )
	response=llm.invoke(new_prompt_template.format(template_userInput=query,template_ageoption=age_option,template_tasktype_option=tasktype_option))

```

### Memorizing Conversations 

Memory+Module.ipynb

```
conversation = ConversationChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory()
)
conversation("Good morning AI!")
conversation("My name is sharath!")
conversation.predict(input="What is my name?")
```

### Query  Data from Embeddings
Use OpenAI Embeddings or all-MiniLM-L6-v2 to search from our own data
Use  all-MiniLM-L6-v2
Store in Chroma vector database
Data+Connections+-+Overview.ipynb

```
loader = TextLoader('Sample.txt')
documents = loader.load()
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_kwargs={"k": 2})
docs = retriever.get_relevant_documents("What is the capital of india?")
```

### Get Answers from PDF Documents

Transform and Store documents in PineCore
Get Answers
MCQ Creator App - Jupyter Notebook\MCQ Creator.ipynb
```
  loader = PyPDFDirectoryLoader(directory)
  embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2") 
  #embeddings = OpenAIEmbeddings(model_name="ada")
  PineconeClient(api_key=PINECONE_API_KEY, environment="gcp-starter")
  index_name="chatbot"
  index = Pinecone.from_documents(docs, embeddings, index_name=index_name)
  similar_docs = index.similarity_search(query, k=k)
```

### Chaining data
Get Data from Google and get results
Chain the data
Utility+Chains+Overview
Generic+Chains+Overview.ipynb

### Querying from CSV

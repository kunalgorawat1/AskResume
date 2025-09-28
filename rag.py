"""
1. Load the pdf file.
2. Extract the text from pdf and split into small chunks.
3. Send the chunks to the embedding model.
4. Save the embeddings to the vector database (chormadb here)
5. Perform similarity search on the vector database to find similar documents.
6. retrieve the similar documents and present them to the user.
"""
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import OneDriveFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

doc_path = "./data/KunalResume.pdf"
model = "llama3.2"

# upload pdf

if doc_path:
    loader = UnstructuredPDFLoader(file_path=doc_path)
    data = loader.load()
    print("DONE.")
else:
    print("Could not upload.")

content = data[0].page_content
# print(content[:10000])

# loading done

# extract text and split into chunks.
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 


# split and chunk
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
chunks = text_splitter.split_documents(data)
print("splitting compelted")

# print(f"Number of chunks: {len(chunks)}")
# print(f"chunks: {chunks[0]}")

# get the embedding model (another llm)
import ollama
ollama.pull("nomic-embed-text")

vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model="nomic-embed-text"),
    collection_name="resume-rag"
)

print("done adding embeddings to vector db")

# Retrieval

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

llm = ChatOllama(model=model)
PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an assistant, your task is to summarize the resume and respond to queries of the interviewer
    and create few question that the interviewer can ask the candidate based on their resume. The intertion is to cover
    all the experiences the candidate has on their resume, generate realistic, easy to understand questions.
    generate level of questions as well, categorize them in easy, medium and hard on the technologies and experience of
    the candidate.
    Do not respond to irrelevant queries, politely decline to answer if asked.
    response should be in a numbering format.
    Here is the user's Question {question}
    """,

)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm, prompt=PROMPT
)

# rag promt
template = """
    Answer the question based only on the following context
    {context}
    Qeustion: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

res = chain.invoke(input=("Generate the questions I as an interviewer can ask the candidate."))

print(res)
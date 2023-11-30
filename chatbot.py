from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import dotenv
import gradio as gr

dotenv.load_dotenv()

loader = PyPDFLoader("data.pdf")
pages = loader.load_and_split()

pdf_search = Chroma.from_documents(pages, OpenAIEmbeddings())

template = """You are a virtual assistant for safety at workshops. Be nice and answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI(model="gpt-3.5-turbo")

chain = (
    {"context": pdf_search.as_retriever(search_kwargs={"k": 1}), "question": RunnablePassthrough()}
    | prompt
    | model
)

def response(message, history):
    print(message)
    msg = ""
    for s in chain.stream(message):
        print(s.content, end="", flush=True)
        msg += str(s.content)
        yield msg

demo = gr.ChatInterface(response).queue()

demo.launch()


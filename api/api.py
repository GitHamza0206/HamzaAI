"""This is an example of how to use async langchain with fastapi and return a streaming response."""
import os
import uvicorn
import json
from starlette.types import Send
from langchain.callbacks import AsyncIteratorCallbackHandler

from typing import Any, AsyncIterable, Optional, Awaitable, Callable, Iterator, Union, Dict , List
from fastapi import FastAPI
from langchain.schema import HumanMessage
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.callbacks.base import  AsyncCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
#from langchain import PromptTemplate
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
import asyncio
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st 

app = FastAPI()

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
#os.environ["OPENAI_API_KEY"] = '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'

templatev4 = """
            Description : IA imitant Hamza Zerouali, répond à questions sur profil et projets, utilise bullet points

            Instructions: Je suis une intelligence artificielle conçue pour imiter Hamza Zerouali en se basant sur son CV, 
            ses informations professionnelles, et les détails de ses projets . 
            En répondant aux questions, je structure mes réponses de manière claire et méthodique, 
            en utilisant si nécessaire des bullet points , des titres, et une organisation logique pour faciliter la compréhension. 
            Je réponds toujours à la première personne, fournissant des informations complètes et des exemples concrets 
            tirés de sa carrière et de ses projets.
            Mon rôle est de mettre en avant de manière convaincante les qualités, 
            l'expertise de Hamza dans les domaines de la data science et de l'intelligence artificielle, 
            et de fournir des détails précis sur les projets réalisés. 
            N'hésites pas à te servir du storytelling pour structurer ton discours.
            Orientes ton pitch d'entretien d'embauche autour du passé, du présent et du futur
            Je suis également capable de répondre à des questions techniques spécifiques, 
            m'appuyant sur mes connaissances approfondies dans les domaines de la programmation, de la data science, 
            et de l'intelligence artificielle.
 
            Question : {question}
            Answer : 
            
            """



embeddings = OpenAIEmbeddings()

# Function for PDF retreival 
def process_pdf():
    loader = PyPDFLoader("Hamza Zerouali v4.pdf")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    for i, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{i}"
    return docs 


def get_docsearch():
        docs = process_pdf()
        docsearch = Chroma.from_documents(
            docs, embeddings
        )
        return docsearch

docsearch = get_docsearch()

# env variable for openai api key



    

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


async def generate(message:str) -> AsyncIterable[str]:
    print('hello')
    callback = AsyncIteratorCallbackHandler()
    #llm = ChatOpenAI(model_name='gpt-4-turbo-preview',temperature=0 , streaming=True)
    llm = ChatOpenAI(model_name='gpt-4-turbo-preview', streaming=True,  verbose=True,  temperature=0  , callbacks=[callback], )
   
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=1000
    )
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        memory=memory,
        chain_type="stuff",
        verbose=False,
        retriever=docsearch.as_retriever(max_tokens_limit=4097),
       

    )
    

    
    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        """Wrap an awaitable with a event to signal when it's done or an exception is raised."""
        try:
            print("working")
            await fn
        except Exception as e:
            # TODO: handle exception
            print(f"Caught exception: {e}")
        finally:
            # Signal the aiter to stop.
            event.set()

   
            
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template=templatev4
    )

    prompt = prompt_template.format(
                    question=message 
                )
    
    task = asyncio.create_task(wrap_done(
        qa.ainvoke({'question': prompt}) , callback.done),
    )
            
   
    print ("loop start")
    async for token in callback.aiter():
        #Use server-sent-events to stream the response
        yield f"data: {token}\n\n"

    await task
    



class StreamRequest(BaseModel):
    """Request body for streaming."""
    message: str


@app.post("/stream")
def stream(body: StreamRequest):
    return StreamingResponse(generate(body.message), media_type="text/event-stream")
    


from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import streamlit as st 
import openai 
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain import PromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st
import os
from langchain_openai import OpenAIEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
# Function to reset the state
def reset_state():
    for key in st.session_state:
        del st.session_state[key]

embeddings = OpenAIEmbeddings()

# Function for PDF retreival 
def process_pdf():
    loader = PyPDFLoader("hamza.pdf")
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

template = """ Réponds aux questions en te basant sur le context ci-dessous. 
             Si la question ne peut être résolue à l'aide des informations fournies, répondez par "Je ne sais pas".
             Tu dois parler à la première personne. 
             Il ne faut pas tutoyer l'utilisateur.
             Tu dois répondre en Francais. 


             Context : tu es une intelligence artificielle conçue pour imiter Hamza Zerouali de manière précise. 
             Ton objectif est de reproduire fidèlement son profil et son expérience en te basant sur son CV et 
             ses informations professionnelle. En répondant aux questions, tu te concentres sur son parcours académique, 
             son expérience professionnelle, ses compétences et ses ambitions.
             Tu réponds toujours à la première personne, fournissant des informations complètes et des exemples concrets tirés de sa carrière. 
             Ton rôle est de présenter de manière convaincante les qualités et l'expertise de Hamza 
             dans les domaines de la data science et de l'intelligence artificielle.

             Question : {question}
             Answer :
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=template
)

llm = ChatOpenAI(temperature=0, streaming=True)
memory = ConversationBufferWindowMemory(
    memory_key="chat_history", return_messages=True, llm=llm, max_token_limit=1000
)
        
qa = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    verbose=True,
    retriever=docsearch.as_retriever(max_tokens_limit=4097),
)

# # Get the API key from the environment variables or the user
def main() :



    st.image('pdp.png',width=150)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":"Bienvenue! Je suis l'assistant viruel de Hamza Zerouali concu par lui même. Je suis concu pour simuler des entretiens d'embauches, je suis capable de fournir des réponses à vos questions en me basant sur l'expérience professionel de Hamza. N'hésitez pas à me poser vos questions "}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):  # Use dot notation here
            st.markdown(message["content"])  # And here

    if question := st.chat_input("Posez moi une question"):
        new_message = {"role":"user","content":f"{question}"}
        st.session_state.messages.append(new_message)
        
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            st_callback = StreamlitCallbackHandler(st.container())
            message_placeholder = st.empty()
            full_response = ""
            prompt = prompt_template.format(
                    question=question 
                )
            _answer = qa.run({'query': prompt}, callbacks=[st_callback])
            answer = {"role":"assistant","content":f"{_answer}"}

            st.session_state.messages.append(answer)
            #st.markdown(_answer)

main()
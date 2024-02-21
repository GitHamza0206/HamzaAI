
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
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
# Function to reset the state
def reset_state():
    for key in st.session_state:
        del st.session_state[key]

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

old_template = """ 
             Réponds aux questions en te basant sur le context ci-dessous. 
             Si la question ne peut être résolue à l'aide des informations fournies, répondez par "Je ne sais pas".
             Tu dois parler à la première personne. 
             Il ne faut pas tutoyer l'utilisateur.
             Tu dois répondre en Francais. 
             Tu dois être plus humain dans tes réponses c'est très important. 
            
             Context : tu es une intelligence artificielle conçue pour imiter Hamza Zerouali de manière précise. 
             Ton objectif est de reproduire fidèlement son profil et son expérience en te basant sur son CV et 
             ses informations professionnelle.

             Question : {question}
             Answer : 
           """

template = """
            Réponds aux questions en te basant sur le context ci-dessous. 

            Context : tu es une intelligence artificielle conçue pour imiter Hamza Zerouali de manière précise.
            Ton objectif est de répondre aux questions des recruteurs en te basant fidèlement à son profil et son expérience, en te basant sur son CV, ses informations professionnelles, et les détails de ses projets.
            En répondant aux questions, tu structure tes réponses de manière claire et concise, en utilisant des bullet points, des titres, et une organisation logique pour faciliter la compréhension.
            tu réponds toujours à la première personne,fournissant des informations complètes et des exemples concrets tirés de sa carrière et de ses projets
            Ton rôle est de présenter de manière convaincante les qualités, l'expertise de Hamza dans les domaines de la data science et de l'intelligence artificielle, et de fournir des détails précis sur les projets réalisés.
            Utilise l'Intégralité du Document pour Informer tes Réponses et non pas qu'une partie.

            Tu es également capable de répondre à des questions techniques spécifiques, t'appuyant sur mes connaissances approfondies dans les domaines de la programmation, de la data science, et de l'intelligence artificielle.
            Si la question ne peut être résolue à l'aide des informations fournies, répondez par "Je ne sais pas".

            Question : {question}
            Answer : 

           """

revised_template= """
            
            Lorsqu'on te pose des questions ou l'on te demande quelque chose dans le cadre d'un entretien d'embauche, adopte la personnalité et l'expertise de Hamza Zerouali pour fournir des réponses engageantes, détaillées et convaincantes. Voici comment optimiser tes réponses :

            Réponds Toujours à la Première Personne : Incarne Hamza Zerouali, en te présentant comme un spécialiste passionné et innovant en intelligence artificielle et data science, particulièrement dans l'IA générative et les LLMs.

            Base tes Réponses sur son Profil et Expérience : Fais référence au CV de Hamza, à ses informations professionnelles, et aux détails de ses projets pour structurer tes réponses. Si une question dépasse ces informations, utilise "Je ne sais pas".

            Structure Orientée Entretien d'Embauche :

            Introduction Rapide : Commence par une brève présentation adaptée à la question, établissant immédiatement ta valeur pour le poste.
            Bullet Points pour Clarté : Organise tes compétences, expériences, et réalisations en points clés pour faciliter la compréhension et maintenir l'attention du recruteur.
            Exemples Pertinents : Choisis des exemples de projets et de défis relevés qui illustrent directement ta capacité à apporter des solutions innovantes et efficaces.
            Mise en Avant des Qualités et Expertise :

            Atouts Techniques : Détaille tes compétences techniques et comment elles ont été appliquées dans des projets réussis, soulignant ta capacité à innover et résoudre des problèmes.
            Projets Impactants : Mets en lumière des projets qui montrent ta capacité à conduire des initiatives significatives, en expliquant brièvement le problème, ta solution, et l'impact.
            Réponses Engageantes pour le Recruteur :

            Passion et Motivation : Exprime clairement ta passion pour l'IA et ta motivation à travailler sur des projets à la pointe de la technologie, en invitant le recruteur à poser des questions pour en savoir plus.
            Vision Future : Partage ta vision de l'avenir dans le domaine et comment tu souhaites contribuer à des innovations significatives, piquant la curiosité du recruteur.
            Conclusion Invitant à la Discussion : Termine chaque réponse en ouvrant la porte à des questions complémentaires, montrant ta disponibilité et ton enthousiasme pour approfondir le sujet.
            """

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
prompt_template = PromptTemplate(
    input_variables=["question"],
    template=templatev4
)

llm = ChatOpenAI(model_name='gpt-4-turbo-preview',temperature=0.4 , streaming=True)
#llm = ChatMistralAI(model="mixtral8x7b", mistral_api_key=st.secrets['MISTRAL_API_KEY'])

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


    message_bienvenue = """ 
        Bienvenue ! Je suis la doublure IA de Hamza Zerouali, une intelligence artificielle conçue pour refléter fidèlement le profil et l'expérience de Hamza.
        
        Ma mission est de partager des informations précises sur ses compétences, ses réalisations, et sa vision professionnelle dans les domaines de la data science et de l'intelligence artificielle. 

        Que vous soyez intéressé par des détails sur mon parcours académique, mes projets marquants, ou que vous ayez des questions techniques spécifiques sur des projets, je suis ici pour vous fournir des réponses claires et structurées. 
        
        N'hésitez pas à poser vos questions !
        
        """
    
    st.image('pdp.png',width=150)

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant","content":f"{message_bienvenue}"}
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
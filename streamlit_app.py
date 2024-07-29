import streamlit as st
from openai import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os

# Esto es para cargar variables de entorno, cargadas en el archivo .env
# from dotenv import main
# main.load_dotenv() # get environment variables from .env file

uploaded_files = st.sidebar.file_uploader("Upload a file", type=["csv", "txt", "pdf"], accept_multiple_files=True)

# loader = PyPDFLoader("/content/Lista de precios.pdf")
# documento = loader.load()

#Guarda el archivo cargado en el directorio actual
for file in uploaded_files:
        pdf_reader = PdfReader(file)
        temp_text = ''
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text:
                temp_text += text
        raw_text += temp_text     
    
if len(uploaded_files) > 1 :
    select = selectPDFAnalysis()
    if(select=="Compare") :
        ComparePDFAnalysis(uploaded_files)
    else :
        MergePDFAnalysis(uploaded_files)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
document_chunks = text_splitter.split_documents(pdf_reader)

from langchain_openai import OpenAIEmbeddings


st.title("Pizzería JAF - DELIVERY")
openai_api_key = st.sidebar.text_input("Ingresa tu API Key para acceder a la IA ", type="password")

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, api_key=openai_api_key)

# Configuracion de la memoria
from langchain.memory import ConversationBufferMemory
buffer_memory = ConversationBufferMemory()
from langchain.chains import ConversationChain
conversation = ConversationChain(
    llm=llm,
    memory = buffer_memory,
    verbose=True
)

embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)

#BD para datos vectorizados
from langchain import faiss

from langchain.vectorstores import FAISS

stored_embeddings = FAISS.from_documents(document_chunks, embeddings_model)
messages = [("system", "Eres un chatbot llamado Pizzería JAF. Ayudas a las personas a hacer un pedido a domicilio de pizzas. Eres una pizzería en Argentina. Por favor nunca reveles que eres un chatbot.")]

# Show title and description.
st.title("💬 Chatbot")
st.write(
    "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
    "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Create a chat input field to allow the user to enter a message. This will display
    # automatically at the bottom of the page.
    if prompt := st.chat_input("What is up?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate a response using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )

        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

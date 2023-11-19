import streamlit as st  # Import the Streamlit library and alias it as 'st'

from dotenv import load_dotenv  # Import the load_dotenv function from the dotenv library
from PyPDF2 import PdfReader  # Import PdfReader class from PyPDF2 library
from langchain.text_splitter import CharacterTextSplitter  # Import CharacterTextSplitter class from langchain.text_splitter module
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings  # Import OpenAIEmbeddings and HuggingFaceInstructEmbeddings classes from langchain.embeddings module
from langchain.vectorstores import FAISS  # Import FAISS class from langchain.vectorstores module
from langchain.chat_models import ChatOpenAI  # Import ChatOpenAI class from langchain.chat_models module
from langchain.memory import ConversationBufferMemory  # Import ConversationBufferMemory class from langchain.memory module
from langchain.chains import ConversationalRetrievalChain  # Import ConversationalRetrievalChain class from langchain.chains module
from htmlTemplates import css, bot_template, user_template  # Import css, bot_template, and user_template from htmlTemplates module
from langchain.llms import HuggingFaceHub  # Import HuggingFaceHub class from langchain.llms module

def get_pdf_text(pdf_docs):  # Define a function get_pdf_text that takes in pdf_docs as a parameter
    text = ""  # Initialize an empty string variable called 'text'
    for pdf in pdf_docs:  # Iterate through each PDF document in pdf_docs
        pdf_reader = PdfReader(pdf)  # Create a PdfReader object for the current PDF document
        for page in pdf_reader.pages:  # Iterate through each page in the PDF document
            text += page.extract_text()  # Extract text from the current page and append it to 'text'
    return text  # Return the concatenated text from all pages of PDFs

# Define a function get_text_chunks that takes in text as a parameter
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(  # Create a CharacterTextSplitter object
        separator="\n",  # Set the separator as newline
        chunk_size=1500,  # Set the chunk size as 1500 characters
        chunk_overlap=250,  # Set the overlap between chunks as 250 characters
        length_function=len  # Use the len function to determine length
    )
    chunks = text_splitter.split_text(text)  # Split the text into chunks using the text_splitter object
    return chunks  # Return the chunks of text

# Define a function get_vectorstore that takes in text_chunks as a parameter
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()  # Create an OpenAIEmbeddings object
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)  # Create a vector store using FAISS from the text chunks with embeddings
    return vectorstore  # Return the vector store

# Define a function get_conversation_chain that takes in vectorstore as a parameter
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()  # Create a ChatOpenAI object
    memory = ConversationBufferMemory(  # Create a ConversationBufferMemory object
        memory_key='chat_history',  # Set the memory key as 'chat_history'
        return_messages=True  # Set return_messages parameter as True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(  # Create a ConversationalRetrievalChain using from_llm method
        llm=llm,  # Pass the ChatOpenAI object
        retriever=vectorstore.as_retriever(),  # Use the vectorstore as retriever
        memory=memory  # Pass the ConversationBufferMemory object
    )
    return conversation_chain  # Return the conversational chain

# Define a function handle_userinput that takes in user_question as a parameter
def handle_userinput(user_question):
    if st.session_state.conversation:  # Check if conversation state exists in session
        try:
            response = st.session_state.conversation({'question': user_question})  # Generate response using conversation chain
            st.session_state.chat_history = response['chat_history']  # Store chat history in session state

            for i, message in enumerate(st.session_state.chat_history):  # Iterate through chat history
                if i % 2 == 0:  # Check if index is even
                    if hasattr(message, 'link'):  # Check if message has 'link' attribute
                        st.markdown(user_template.replace(  # Display user template with link if present
                            "{{MSG}}", f'<a href="{message.link}" target="_blank">{message.content}</a>'), unsafe_allow_html=True)
                    else:
                        st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # Display user template content
                else:
                    st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)  # Display bot template content
        except Exception as e:
            st.error("An error occurred. Please contact the admin assistant at admis@sta.uwi.edu.")  # Display error message in case of an exception
    else:
        st.write("Please upload and process your documents first.")  # Display message if conversation is not set up yet

# Define the main function
def main():
    load_dotenv()  # Load environment variables
    st.set_page_config(page_title="UWI Chat-Bot", page_icon=":books:")  # Set Streamlit page configuration
    st.write(css, unsafe_allow_html=True)  # Display CSS styles

    if "conversation" not in st.session_state:  # Check if conversation is not in session state
        st.session_state.conversation = None  # Initialize conversation as None
    if "chat_history" not in st.session_state:  # Check if chat history is not in session state
        st.session_state.chat_history = None  # Initialize chat history as None

    st.header("UWI Chat-Bot :books:")  # Display header
    user_question = st.text_input("Ask a question about your documents:")  # Input for user question
    if user_question:  # Check if user question is provided
        handle_userinput(user_question)  # Handle user input

    with st.sidebar:  # Create a sidebar in the UI
        st.subheader("Your documents")  # Display subheader in sidebar
        pdf_docs = st.file_uploader(  # File uploader for PDF documents
            "Upload your PDFs here and click on 'Process'. Disclaimer: Please submit only University documents to be processed here.  ",
            accept_multiple_files=True)  # Allow multiple files to be uploaded
        if st.button("Process"):  # Check if 'Process' button is clicked
            with st.spinner("Processing"):  # Display spinner during processing
                raw_text = get_pdf_text(pdf_docs)  # Extract text from uploaded PDFs
                text_chunks = get_text_chunks(raw_text)  # Split text into chunks
                vectorstore = get_vectorstore(text_chunks)  # Create vector store
                st.session_state.conversation = get_conversation_chain(vectorstore)  # Initialize conversation chain

if __name__ == '__main__':  # Check if the script is executed as the main program
    main()  # Execute the main function

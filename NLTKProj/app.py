import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


#Defining a function named 'get_pdf_text' that takes a list of 'pdf_docs' as an input
def get_pdf_text(pdf_docs):
    #Initialising an empty string 'text' that stores the extracted text.
    text = ""
    #For loop used to iterate through the list of 'pdf_docs'.
    for pdf in pdf_docs:
        #Creates a PdfReader object for the current PDF document.
        pdf_reader = PdfReader(pdf)
        #For loop used to iterate through the pages of the PDF document.
        for page in pdf_reader.pages:
            #Extract the text from the current page and append it to the 'text' string.
            text += page.extract_text()
            #Return the concatenated 'text' from all the PDF documents.
    return text

#Defining a function named 'get_text_chunks' that takes a 'text' as an input.
def get_text_chunks(text):
    #Created an instance of 'CharacterTextSplitter' and configured it.
    text_splitter = CharacterTextSplitter(
        #this splits the text at the line breaks.
        separator="\n",
        #Create chunks of the text up to 1000 characters
        chunk_size=1000,
        #Allows an overlap of 200 characters between chunks.
        chunk_overlap=200,
        #Created an instance 'len' where the function is used to calculate the length of the text.
        length_function=len
    )
    #Use the 'split_text' method of the 'text_splitter' to split the input 'text' into chunks
    chunks = text_splitter.split_text(text)
    #Returns the list of 'chunks'.
    return chunks

# This function handles the embedding of our text chunks which were generated from the PDF.Embedding are numerical vector representations of our chunks of 
# text where each word or phrase is represented as a vector of numbers. Words that are similar in meaning or context are stored in vectors which are close 
# togeher in vector space which makes it easier for our Chatbot to determine an appropriate answer based on the data from the PDF. The embedding in this
# code is done using OpenAI's embedding model.
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings() 
    # Some exception handling is done to check if the vector store has been successfully created
    try:
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)    # FAISS is used as a data base to create a vector store so that we can store
        if vectorstore is None:                                                    # our embeddings. FAISS runs locally so all the data is stored on our machine    
            st.error("Vectorstore creation failed: returned None")                 # as opposed to a cloud. 
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None





# The function below allows the user to have a conversation with the chatbot using ChatOpenAI llm, it also creates a memory store so that a user can ask a follow 
# up question from a previous question and Chatbot will be able to know the context of that question based on previous questions and responses.
def get_conversation_chain(vectorstore): 
    llm = ChatOpenAI()
    
    # This is the instance at which the conversation memory is created.
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain # This returns the conversation chain 


def handle_userinput(user_question):
    if st.session_state.conversation:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                if hasattr(message, 'link'):
                    st.markdown(user_template.replace(
                        "{{MSG}}", f'<a href="{message.link}" target="_blank">{message.content}</a>'), unsafe_allow_html=True)
                else:
                    st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        # Handle the case where the conversation is not set up yet
        st.write("Please upload and process your documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="UWI Chat-Bot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("UWI Chat-Bot :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'. Disclaimer: Any information uploaded is at your risk. Please refrain from entering personal information.  ", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == '__main__':
    main()

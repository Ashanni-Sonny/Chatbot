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


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
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
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()

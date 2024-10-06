import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from groq import Groq
import sys
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import sqlite3
from langchain_cohere import CohereEmbeddings

def get_embedding_function():
    embeddings = CohereEmbeddings(cohere_api_key="yLa4P1FNzncjNN90YZGTukQciYi2NtZs85WiavFY")
    return embeddings

groq_client = Groq(api_key="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")
# groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
CHROMA_PATH = "chroma"

sys_msg = (
    'You are a College  assistant named SmartYIT who will assist queries on Yenepoya Institute of technology college . Your job is to answer user query based on the college details  context '
    'you will be given relevant chunks from the college data using retrieval augmented generation '
    'consider all the text in the paragraph and answer according to the user query '
    'dont make up your own answers provide only anserws based on the college  data provided  '
    'if there is no information to user query(for example non college  related queries) in  colllege details provided just say it to the user '
    'Use all of the context of this conversation so your response is relevant to the conversation. '
    'Make  your responses clear and concise,your response will be read by students who want to join this college  make sure you give response in friendly manner '
    "make the users join this college "
    " Mr. Rohith is our new skill development co-ordinator at YIT "
)

convo = [
    {'role': 'system', 'content': sys_msg}
]

def query_rag(query_text: str, context: str = None):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    # st.write(embedding_function.embed_query(query_text))
    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=2)
    #st.write(results)
    # Filter out results with None page_content
    valid_results = [(doc, score) for doc, score in results if doc.page_content and doc.page_content.strip()]

    if not valid_results:
        return "No relevant information found in the textbook.", context

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in valid_results])

    if context:
        context_text = context + "\n\n---\n\n" + context_text

    prompt = f'USER PROMPT: {query_text}\n\n  textbook CONTEXT: {context_text}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content, context_text

def main():
    st.markdown("<h1 style='font-family: fantasy;'>SMART YIT 🤖</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .stApp {
        background-color: #e7e5d8; 
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
    """
    <style>
    .stChatInputContainer {
        background-color: #e7e5d8 ;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    
    
    # Initialize chat history and context memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "context" not in st.session_state:
        st.session_state.context = None

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is your question?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Retrieve context if available
            context = st.session_state.context
            
            with st.spinner("Thinking..."):
                # Pass both prompt and context to query function
                response, new_context = query_rag(prompt, context)
            
            message_placeholder.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Update context memory
        st.session_state.context = new_context

if __name__ == "__main__":
    main()

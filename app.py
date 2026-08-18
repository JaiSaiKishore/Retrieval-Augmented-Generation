import streamlit as st
import time

# Import the graph we built in the previous step
from src.graph_agent import app_graph

st.set_page_config(page_title="AI Financial Analyst", layout="wide")

st.title("🤖 Enterprise RAG System")
st.markdown("### Powered by Llama 3 (Groq), Hybrid Search & Cross-Encoder Re-ranking")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your financial document..."):
    
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 2. Display Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🧠 Agent is Thinking..."):
            try:
                # --- THE MAGIC HAPPENS HERE ---
                inputs = {"question": prompt}
                result = app_graph.invoke(inputs)
                
                answer = result['answer']
                context_docs = result['context']
                
                # Show the Answer
                message_placeholder.markdown(answer)
                
                # Advanced: Show "Evidence" (Good for Interview Demo)
                with st.expander("🔍 View Retrieved Evidence (Source Chunks)"):
                    if context_docs:
                        for i, doc in enumerate(context_docs):
                            st.markdown(f"**Source {i+1} (Relevance Score: High):**")
                            st.caption(doc.page_content[:300] + "...") # Show first 300 chars
                            st.divider()
                    else:
                        st.warning("No relevant documents found in the database.")

                # Save history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
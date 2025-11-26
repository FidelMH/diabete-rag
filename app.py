import streamlit as st
from documents import EmbedderRag
from llm import LlmManager

# Page configuration
st.set_page_config(
    page_title="Diabetes Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical design
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTextInput > div > div > input {
        background-color: white;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """
    Initialize RAG system (LLM + Index).
    This function is cached to avoid reloading the system on each interaction.
    """
    with st.spinner("Initializing RAG system..."):
        # Initialize LLM
        llm_manager = LlmManager()

        # Build or load index
        embedder = EmbedderRag(input_path="./documents")
        index = embedder.build_or_load_index()

        # Create query engine
        query_engine = index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )

        # Get configuration info
        config = {
            'llm_model': llm_manager.get_config()['model_name'],
            'embedding_provider': embedder.embedding_manager.provider,
            'embedding_model': embedder.embedding_manager.embed_model.model_name
        }

    return query_engine, config


def main():
    # Title and description
    st.title("🩺 Medical Assistant - Diabetes")
    st.markdown("""
    Ask your questions about diabetes and get answers based on medical documents.
    """)

    # Sidebar with information
    with st.sidebar:
        st.header("📋 Information")
        st.markdown("**Diabetes RAG System**")
        st.markdown("---")

        st.markdown("**🤖 Configuration:**")
        st.markdown(f"- **LLM:** {config['llm_model']}")

        if config['embedding_provider'] == 'azure':
            st.markdown(f"- **Embeddings:** Azure OpenAI ({config['embedding_model']})")
        else:
            st.markdown(f"- **Embeddings:** Local ({config['embedding_model']})")

        st.markdown("- **Framework:** LlamaIndex")
        st.markdown("- **Documents:** Medical documents on diabetes")

        st.markdown("---")

        st.markdown("**Instructions:**")
        st.markdown("""
        1. Type your question in the field below
        2. Press Enter or click Send
        3. Wait for the response based on documents
        """)

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.rerun()

    # Initialize RAG system
    try:
        query_engine, config = initialize_rag()
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        st.stop()

    # Initialize message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Question input area
    if prompt := st.chat_input("Ask your question about diabetes..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = query_engine.query(prompt)
                    response_text = str(response)

                    # Display response
                    st.markdown(response_text)

                    # Display sources in an expander
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        with st.expander("📚 View Sources"):
                            for idx, node in enumerate(response.source_nodes, 1):
                                st.markdown(f"**Source {idx}** (Score: {node.score:.3f})")
                                st.text(node.text[:300] + "..." if len(node.text) > 300 else node.text)
                                st.divider()

                    # Add response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    # Medical disclaimer
    st.info("""
        **Warning:** This chatbot is provided for informational purposes only and does not constitute medical advice or a medical device.
        It does not replace a consultation, diagnosis, or treatment provided by a qualified healthcare professional.
        For any health questions, always consult your doctor.
    """)


if __name__ == "__main__":
    main()

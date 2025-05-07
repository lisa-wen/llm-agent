import streamlit as st
from typing import Annotated
from langchain_core.messages import SystemMessage, AIMessage, RemoveMessage
from langgraph.graph import StateGraph, START, END
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
import openai
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from faq import FaqTool
from recommender_tool import RecommenderTool
from sparql_tool import SPARQLTool
from utils import get_index
from IPython.display import Image
from langchain_chroma import Chroma


# Speichere Session-Nachrichten
class State(TypedDict):
    messages: Annotated[list, add_messages]


# Chatbots mit Tools
def chatbot(state: State):
    return {"messages": [st.session_state.llm_with_tools.invoke(state["messages"])]}


# Initialisierung der Session-Variablen
if 'graph_initialized' not in st.session_state:
    st.session_state.graph_initialized = False
    st.session_state.messages = []
# Start-Initialisierung des Workflow-Graphen
if not st.session_state.graph_initialized:
    # API-Key laden
    load_dotenv()
    openai.api_key = os.getenv('OPENAI_API_KEY')
    # Suchindex erstellen
    searcher = get_index("index")
    # Tools einrichten
    search = DuckDuckGoSearchResults(max_results=5)
    recommender_tool = RecommenderTool(searcher)
    sparql_tool = SPARQLTool()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    faq_tool = FaqTool(vectors_store)
    #tools = [search, recommender_tool, sparql_tool]
    tools = [sparql_tool]
    #tools = [faq_tool]
    # LLM einrichten
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.5,
                     max_tokens=5000)
    st.session_state.llm_with_tools = llm.bind_tools(tools)
    # Graph (Workflow) erstellen
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge(
        "tools",
        "chatbot",
    )
    memory = MemorySaver()
    # Workflow-Graph als Session-Variable speichern
    st.session_state.graph = graph_builder.compile(checkpointer=memory)
    # Graph-Visualisierung speichern
    try:
        graph_image = Image(st.session_state.graph.get_graph().draw_mermaid_png())
        with open('img/langgraph_diagram.png', 'wb') as f:
            f.write(graph_image.data)
    except Exception:
        pass

    st.session_state.graph_initialized = True

config = {"configurable": {"thread_id": "1"}}


# Streamlit-Interface
def main():
    st.title("Search Assistant")

    # Chat-Historie
    for message in st.session_state.messages:
        if message["role"] in ["user", "assistant"]:
            if message["role"] == "user":
                # with st.chat_message("user"):
                with st.chat_message("user", avatar="img/girl.png"):
                    st.markdown(message["content"])
            else:
                # with st.chat_message("assistant"):
                with st.chat_message("assistant", avatar="img/frankenstein.png"):
                    st.markdown(message["content"])

    # Nutzerinput verarbeiten
    if prompt := st.chat_input("Ask me anything"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
        # with st.chat_message("user", avatar="img/girl.png"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
        # with st.chat_message("assistant", avatar="img/frankenstein.png"):
            user_messages = [message['content'] for message in st.session_state.messages if
                             message['role'] == 'user']

            # Eingehende Prompts in den Workflowgraphen integrieren
            results = st.session_state.graph.stream({"messages": user_messages},
                                                    config,
                                                    stream_mode="updates")

            # Ergebnisse der Workflowverarbeitung ausgeben
            for result in results:
                #print(result)
                for node_name, node_output in result.items():
                    # Überprüfe, ob der aktuelle Workflowknoten ein Chatbot-Knoten ist
                    if node_name == "chatbot":
                        # Überprüfe, ob 'messages' existiert und nicht leer ist
                        if 'messages' in node_output and node_output['messages']:
                            # Überprüfe, ob die letzte Nachricht 'content' enthält
                            last_message = node_output['messages'][-1]
                            if hasattr(last_message,'content') and last_message.content:
                                # Gib die letzte AI-/Chatbotnachricht aus
                                response = last_message.content
                                st.markdown(response)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                # Unterbrich die Schleife
                                # (Annahme: Es gibt nur eine relevante Nachricht pro Worklowschritt
                                break


if __name__ == "__main__":
    main()

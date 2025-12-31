import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from src.helper import get_embeddings
from src.prompt import system_prompt

# Flask app
application = Flask(__name__)
load_dotenv()

# Env vars
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

INDEX_NAME = "medical-chatbot-384"

# 🔹 Low-RAM embeddings
embeddings = get_embeddings()

# 🔹 Pinecone (NO ingestion here)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# 🔹 LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

# 🔹 History-aware retriever
contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", "Rephrase the question into a standalone medical query."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

retriever = create_history_aware_retriever(
    llm,
    vectorstore.as_retriever(search_kwargs={"k": 3}),
    contextualize_prompt
)

# 🔹 QA chain
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

@application.route("/")
def index():
    return render_template("chat.html")

@application.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    history_list = request.form.getlist("history[]")

    chat_history = []
    for i in range(0, len(history_list), 2):
        if i + 1 < len(history_list):
            chat_history.append(HumanMessage(content=history_list[i]))
            chat_history.append(AIMessage(content=history_list[i + 1]))

    response = rag_chain.invoke({
        "input": msg,
        "chat_history": chat_history
    })

    return response.get("answer", "Sorry, I couldn’t find an answer.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore

from src.prompt import system_prompt
# from src.helper import download_embeddings   # ✅ REQUIRED
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ---------------- Flask App ----------------
application = Flask(__name__)

# ---------------- ENV ----------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GROQ_API_KEY:
    raise ValueError("Missing API keys")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Lightweight embeddings for Render (NO memory issue)
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001"
)


# ---------------- PINECONE ----------------
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-chatbot",
    embedding=embeddings
)

# ---------------- LLM ----------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4
)

# ---------------- HISTORY AWARE RETRIEVER ----------------
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rephrase the question using chat history if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm,
    docsearch.as_retriever(),
    contextualize_q_prompt
)

# ---------------- QA CHAIN ----------------
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# ---------------- ROUTES ----------------
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

    return response.get("answer", "No answer found")

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    application.run(host="0.0.0.0", port=port)

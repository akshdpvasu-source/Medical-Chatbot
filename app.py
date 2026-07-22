import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeVectorStore


# -------------------------------------------------------
# Load environment variables
# -------------------------------------------------------

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing. "
        "Please add it to your .env file."
    )


# -------------------------------------------------------
# Create Flask application
# -------------------------------------------------------

app = Flask(__name__)


# -------------------------------------------------------
# Create embedding model
# Same embedding model used for your Pinecone index
# -------------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------------------------------
# Connect to existing Pinecone index
# -------------------------------------------------------

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)


# -------------------------------------------------------
# Create retriever
# Same k=6 setting from your trials notebook
# -------------------------------------------------------

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)


# -------------------------------------------------------
# Connect to Ollama
# Same model settings from your trials notebook
# -------------------------------------------------------

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

ChatModel = ChatOllama(
    model="qwen3:1.7b",
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_predict=700,
    num_ctx=4096
)


# -------------------------------------------------------
# Medical chatbot system prompt
# -------------------------------------------------------

system_prompt = """
You are a medical information assistant.

Use only the supplied medical context to answer the user's question.
Do not add unsupported medical facts.

Present information using these Markdown headings:

## Overview
Give a brief explanation of the disease or condition.

## Common Symptoms
List common signs and symptoms.

## Causes and Risk Factors
Explain known causes and risk factors.

## Detection and Diagnosis
Explain how healthcare professionals may detect or diagnose it.

## Prevention and Risk Reduction
Describe appropriate prevention or risk-reduction measures.

## Treatment and Management
Briefly describe general treatment or management information.

## When to Seek Medical Help
Mention warning signs that require professional or emergency care.

Rules:

- Keep the response clear and concise.
- Use bullet points where helpful.
- Do not diagnose the user.
- Do not prescribe medication or give dosages.
- Do not claim prevention is guaranteed.
- If a section is not covered by the context, write:
  "Not specified in the provided medical document."
- For serious symptoms, advise consulting a qualified healthcare
  professional.
- Do not show internal reasoning.

Medical context:

{context}
"""


# -------------------------------------------------------
# Create prompt template
# /no_think prevents Qwen from displaying internal thinking
# -------------------------------------------------------

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "/no_think\n{input}")
    ]
)


# -------------------------------------------------------
# Create RAG chain
# -------------------------------------------------------

question_answer_chain = create_stuff_documents_chain(
    ChatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


# -------------------------------------------------------
# Home page
# -------------------------------------------------------

@app.route("/")
def home():
    return render_template("chat.html")


# -------------------------------------------------------
# Chat API route
# -------------------------------------------------------

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify(
                {
                    "error": "No request data was received."
                }
            ), 400

        question = data.get("question", "").strip()

        if not question:
            return jsonify(
                {
                    "error": "Please enter a medical question."
                }
            ), 400

        # Same invocation used in the trials notebook
        response = rag_chain.invoke(
            {
                "input": question
            }
        )

        answer = response.get(
            "answer",
            "No answer was generated."
        )

        return jsonify(
            {
                "answer": answer
            }
        )

    except Exception as error:
        app.logger.exception(
            "An error occurred while generating the answer."
        )

        return jsonify(
            {
                "error": (
                    "The chatbot could not generate an answer. "
                    "Please check that Ollama is running."
                ),
                "details": str(error)
            }
        ), 500


# -------------------------------------------------------
# Health-check route
# -------------------------------------------------------

@app.route("/health")
def health():
    return jsonify(
        {
            "status": "running",
            "model": "qwen3:1.7b",
            "index": index_name
        }
    )


# -------------------------------------------------------
# Run Flask application
# -------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
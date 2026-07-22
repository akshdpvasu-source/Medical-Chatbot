import os
from typing import Any

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


app = Flask(__name__)

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "medical-chatbot"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv(
    "HUGGINGFACEHUB_API_TOKEN"
)


if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is missing from the environment variables."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing from the environment variables."
    )

if not HUGGINGFACEHUB_API_TOKEN:
    raise ValueError(
        "HUGGINGFACEHUB_API_TOKEN is missing from the "
        "environment variables."
    )

embedding = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)


pinecone_client = Pinecone(
    api_key=PINECONE_API_KEY
)

existing_indexes = pinecone_client.list_indexes().names()

if INDEX_NAME not in existing_indexes:
    raise ValueError(
        f"Pinecone index '{INDEX_NAME}' was not found. "
        f"Available indexes: {existing_indexes}"
    )


# Connect to the already-created Pinecone index.
# This does not upload the PDF or recreate the vectors.

docsearch = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}
)

chat_model = ChatGroq(
    model=GROQ_MODEL,
    temperature=0.2,
    max_tokens=1000,
    groq_api_key=GROQ_API_KEY,
    max_retries=2
)
system_prompt = """
You are a medical information assistant.

Use only the supplied medical context to answer the user's
question. Do not add unsupported medical facts.

Present information using these Markdown headings when they
are relevant to the user's question:

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

- Keep the response clear and useful.
- Use bullet points where helpful.
- Do not diagnose the user.
- Do not prescribe medication or provide dosages.
- Do not claim prevention is guaranteed.
- If information is not covered by the supplied context, say:
  "Not specified in the provided medical document."
- For serious symptoms, advise consulting a qualified healthcare
  professional or emergency service.
- Do not reveal internal reasoning.

Medical context:
{context}
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(
    chat_model,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)



def extract_user_message() -> str:
    """Read the user's message from JSON or form data."""

    if request.is_json:
        data: dict[str, Any] = (
            request.get_json(silent=True) or {}
        )

        message = (
            data.get("message")
            or data.get("input")
            or data.get("question")
            or ""
        )

    else:
        message = (
            request.form.get("message")
            or request.form.get("input")
            or request.form.get("question")
            or ""
        )

    return str(message).strip()


def get_unique_sources(
    context_documents: list[Any]
) -> list[str]:
    """Return unique source names from retrieved documents."""

    sources: list[str] = []
    seen: set[str] = set()

    for document in context_documents:
        source = str(
            document.metadata.get(
                "source",
                "Unknown source"
            )
        )

        if source not in seen:
            seen.add(source)
            sources.append(source)

    return sources

@app.route("/", methods=["GET"])
def home():
    return render_template("chat.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_message = extract_user_message()

    if not user_message:
        return jsonify(
            {
                "error": "Please enter a medical question."
            }
        ), 400

    try:
        response = rag_chain.invoke(
            {
                "input": user_message
            }
        )

        answer = response.get(
            "answer",
            "The chatbot could not generate an answer."
        )

        context_documents = response.get(
            "context",
            []
        )

        sources = get_unique_sources(
            context_documents
        )

        return jsonify(
            {
                "answer": answer,
                "sources": sources
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
                    "Please try again later."
                ),
                "details": str(error)
            }
        ), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "running",
            "model": GROQ_MODEL,
            "index": INDEX_NAME,
            "embedding": (
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        }
    )

if __name__ == "__main__":
    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
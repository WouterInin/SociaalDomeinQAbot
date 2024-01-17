# https://github.com/Spidy20/Flask_NLP_ChatBot

# Doel: Create_database.py en query_data.py samenvoegen en er een front-end bijmaken
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
from tkinter import *
from tkinter import messagebox
from tkinter import Tk, ttk, Text, Entry, Button, END, PhotoImage, Label, WORD, font, RIDGE, Canvas
import sv_ttk

CHROMA_PATH = "chroma"
DATA_PATH = "data/excel"#"data/books"
os.environ["OPENAI_API_KEY"] = "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"

#def main():
#    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.xlsx")#DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

#if __name__ == "__main__":
#    main()

generate_data_store()

# database maken werkt

# query data
import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import Chroma
#from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#import os

#os.environ["OPENAI_API_KEY"] = "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"

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Geef antwoord op de vraag door alleen onderstaande context te gebruiken:

{context}

---

Geef antwoord op deze vraag op basis van bovenstaande context: {question}
"""


def query(query_text):
    # Create CLI.
    #parser = argparse.ArgumentParser()
    #parser.add_argument("query_text", type=str, help="The query text.")
    #args = parser.parse_args()
    #query_text = args.query_text

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"{response_text}\n\n dit antwoord is gebaseerd op de volgende bron(nen): {list(set(sources))}"
    print(formatted_response)
    return formatted_response


import time
time.clock = time.time
#from chatbot import CB
from flask import Flask, render_template, request

application = Flask(__name__)

@application.route("/")
def home():
    return render_template("index.html")

@application.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(query(userText))
application.run(debug = False)

#Publishen: https://www.freecodecamp.org/news/how-to-build-a-web-application-using-flask-and-deploy-it-to-the-cloud-3551c985e492/
#https://learn.microsoft.com/en-us/azure/app-service/quickstart-python?tabs=flask%2Cwindows%2Cazure-portal%2Cvscode-deploy%2Cdeploy-instructions-azportal%2Cterminal-bash%2Cdeploy-instructions-zip-azcli
#ngrok? maar dan moet mijn local host aanstaan
#https://stackoverflow.com/questions/14525029/display-a-loading-message-while-a-time-consuming-function-is-executed-in-flask voor loading

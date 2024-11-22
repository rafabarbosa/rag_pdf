from fastapi import FastAPI
from pydantic import BaseModel

from query_rag import query_rag
from populate_database import load_documents, clear_database

app = FastAPI()

class Question(BaseModel):
  question: str

@app.post("/question")
async def send_question(question: Question):

  response = query_rag(question.question)

  return {
    "question": question.question,
    "answer": response
  }

@app.post("/populate")
async def populate_database():
  response = load_documents()
  return response

@app.post("/reset")
async def reset_database():
  response = clear_database()
  return response
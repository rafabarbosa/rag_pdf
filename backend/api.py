from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from query_rag import query_rag
from populate_database import load_documents, clear_database

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1",
    "http://0.0.0.0",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
  question: str

@app.post("/question")
async def send_question(question: Question):
  try:
    response = query_rag(question.question)

    return {
      "question": question.question,
      "answer": response.content
    }
  except Exception as error:
    return {
      "message": error
    }

@app.post("/populate")
async def populate_database():
  response = load_documents()
  return response

@app.post("/reset")
async def reset_database():
  response = clear_database()
  return response
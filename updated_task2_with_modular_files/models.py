# models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class Metadata(BaseModel):
    similarity_score: float = Field(..., description="Score representing the similarity between the query and document.")

class RetrievedDocument(BaseModel):
    document_id: str = Field(..., description="Unique identifier for the retrieved document.")
    content: str = Field(..., description="Text content of the retrieved document.")
    metadata: Metadata = Field(..., description="Metadata containing the similarity score of the document.")

class RetrievalResult(BaseModel):
    query: str = Field(..., description="User query for retrieving relevant documents.")
    retrieved_documents: List[RetrievedDocument] = Field(..., description="List of documents retrieved based on the query.")
    retrieval_method: str = Field(default="vector search", description="Method used for document retrieval.")
    generated_response: Optional[str] = Field(None, description="Generated response based on the retrieved documents.")

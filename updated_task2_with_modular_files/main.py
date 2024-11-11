# main.py
import os
from database import DatabaseManager
from embedding_generator import EmbeddingGenerator
from utils import load_text_samples
from models import RetrievalResult, RetrievedDocument, Metadata
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_FILE_PATH = 'documents.pdf'
is_pdf = True

def generate_augmented_response(query, retrieved_items, max_context_length=1000):
    print("Generating augmented response...")
    try:
        retrieved_docs = [
            RetrievedDocument(
                document_id=f"doc_{idx}",
                content=text,
                metadata=Metadata(similarity_score=score)
            )
            for idx, (text, score) in enumerate(retrieved_items)
        ]
        context = " ".join(text for text, _ in retrieved_items[:3])
        if len(context) > max_context_length:
            context = context[:max_context_length]

        client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        prompt = f"""You are an expert in programming languages. Based on the context provided, please answer the following question directly:
        Context: {context}
        Question: {query}"""

        chat_completion = client.chat.completions.create(
            messages=[ 
                {"role": "system", "content": "You are an assistant providing direct answers based on the context."},
                {"role": "user", "content": prompt}
            ],
            model="llama3-8b-8192",
            temperature=0,
            max_tokens=1000
        )
        response = chat_completion.choices[0].message.content.strip()
        print("Response generated successfully.")
        return RetrievalResult(
            query=query,
            retrieved_documents=retrieved_docs,
            generated_response=response
        )
    except Exception as e:
        print("Error generating response:", str(e))
        return RetrievalResult(query=query, retrieved_documents=retrieved_docs, generated_response="Error generating response.")

def main():
    db_manager = DatabaseManager()
    embedding_gen = EmbeddingGenerator()

    print("Loading text samples from PDF...")
    texts = load_text_samples(DATA_FILE_PATH, is_pdf=is_pdf)
    
    print(f"Number of text segments loaded: {len(texts)}")
    for idx, text in enumerate(texts):
        embedding = embedding_gen.generate_embedding(text)
        if embedding is not None:
            text_id = str(idx)
            print(f"Storing embedding for text_id: {text_id}...")
            db_manager.add_embedding_to_db(embedding, text_id=text_id, text_content=text)

    print("Embeddings added to the database.")
    
    while True:
        query_text = input("Enter your query (or type 'stop' to finish): ")
        if query_text.lower() == 'stop':
            print("Stopping the query input...")
            break

        print("Generating query embedding...")
        query_embedding = embedding_gen.generate_embedding(query_text)
        print("Searching for similar items...")
        similar_items = db_manager.search_similar_vectors(query_embedding, top_k=3)
        print("Similar items found:", similar_items)

        result = generate_augmented_response(query_text, similar_items)
        print("Generated Response:", result.generated_response)
        db_manager.save_query_response(query_text, result.generated_response)

    db_manager.close()

if __name__ == "__main__":
    main()

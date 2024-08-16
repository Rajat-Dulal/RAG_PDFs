from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from llama_index.core import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    set_global_service_context,
)
from llama_index.llms.ollama import Ollama
import os

# Initialize FastAPI
app = FastAPI()

# Initialize LLM and service context
llm = Ollama(model="gemma:2b", request_timeout=360.0)

# Initialize service context (we'll do this later after the document upload)
service_context = None
index = None

# Define a model for the input query
class QueryRequest(BaseModel):
    query: str

# Endpoint for uploading PDF and creating the index
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global service_context, index

    try:
        # Save the uploaded file
        file_location = f"./{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Read the PDF file using SimpleDirectoryReader
        documents = SimpleDirectoryReader(input_files=[file_location]).load_data()

        # Initialize service context
        service_context = ServiceContext.from_defaults(
            llm=llm, 
            embed_model="local:BAAI/bge-small-en-v1.5", 
            chunk_size=300
        )
        set_global_service_context(service_context)

        # Parse documents into nodes
        nodes = service_context.node_parser.get_nodes_from_documents(documents)

        # Initialize storage context and create the index
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)
        index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=storage_context, 
            llm=llm
        )

        # Clean up the saved file
        # os.remove(file_location)

        return {"message": "PDF uploaded and index created successfully!"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint for querying the LLM
@app.post("/query/")
async def query_llm(request: QueryRequest):
    global index

    if index is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded yet.")

    query = request.query
    try:
        chat_eng = index.as_chat_engine(similarity_top_k=3, chat_mode='context')
        response = chat_eng.chat(query)
        sources = [
            {"score": node.get_score(), "text": node.text}
            for node in response.source_nodes
        ]
        return {"response": response.response, "sources" : sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the server using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

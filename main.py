from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv() #loads variables from .env into the environment
model = SentenceTransformer("all-MiniLM-L6-v2") #loads pretrained embedding model as model
api_key = os.getenv("GEMINI_API_KEY") #checks if there is a variable by this name, and asks for its value
if not api_key:
    raise ValueError("GEMINI_API_KEY was not found. Check your .env file.")
client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/" #allows the client to talk to gemini server
)

#========================functions====================================================================

def load_pdf_text(file_path): 
    reader = PdfReader(file_path) #reader is an object representing the pdf, which we can work with
    pages_data = []

    print(f"Number of pages: {len(reader.pages)}") #reader.pages gives access to the pages attribute of reader. allows us to access pages by index

    for i, page in enumerate(reader.pages): #loop through pages, assigning each a temp index 
        text = page.extract_text() 

        if text: #python treats 'truthy' non zero objects as true by default
            pages_data.append({"page_number": i+1, "text": text}) #dictionary stores properties of the page
        else:
            print(f"Warning: page {i+1} returned no extractable text")
    
    return pages_data

def chunk_pages(pages_data, file_path, chunk_size = 1000, overlap = 150):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    
    chunks = []
    chunk_id = 1
    step = chunk_size - overlap

    for page in pages_data:
        page_number = page["page_number"]
        text = page["text"]

        for start in range(0, len(text), step): #goes over all text in each page in 1000 char steps
            chunk_text = text[start:(start + chunk_size)]

            if not chunk_text.strip(): #chunk_text.strip() is true if not empty, so this line is true if chunk_text is whitespace
                continue #exits the loop for this iteration

            chunks.append({"page_number": page_number, "chunk_id": chunk_id, "text": chunk_text, "source": file_path}) #maintains dictionary structure

            chunk_id += 1 #ensures unique id number

            if start + chunk_size >= len(text): #this is the end index used, so if this is true, we've reached the end of the text
                break #to avoid repeating the last chunk

    return chunks

def add_embeddings(chunks):

    texts = [chunk["text"] for chunk in chunks] #texts are the text attribute of each chunk in chunks 
    embeddings = model.encode(texts) #model.encode() takes input chunks and converts to vectors

    for chunk, embedding in zip(chunks, embeddings): #zip pairs each chunk with its vector
        chunk["embedding"] = embedding.tolist() #.tolist() is used to convert embeddings (usually numpy arrays) to python lists, easier to inspect
    
    return chunks

def embed_query(query):
    return model.encode(query) #Associates a vector embedding to the query asked by user

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) #simple dot product finding cos(theta)

def build_faiss_index(embedded_chunks):
    embeddings = np.array(
        [chunk["embedding"] for chunk in embedded_chunks], #goes through each chunk, gets embedding and puts it in a numpy array
        dtype = "float32") #each row is an embedding, each column is a dimension (component of the vector), so for N chunks, embeddings.shape = (N, 384)
    faiss.normalize_L2(embeddings) # Each vector has euclidean norm of 1, so cosine similarity is the same as dot product
    dimension = embeddings.shape[1] #should be 384

    index = faiss.IndexFlatIP(dimension) #empty faiss index, expecting vectors of dimension 384, when searched, compare using inner product (dot product)
    index.add(embeddings)
    return index

def save_pipeline(embedded_chunks, index, chunks_path="embedded_chunks.json", index_path="faiss_index.bin"):
    with open(chunks_path, "w") as f: #opens file at this path in write mode. 'with' handles closing the file automatically when done.
        json.dump(embedded_chunks, f) #takes python list of dictionaries, writing as json text into the file
    faiss.write_index(index, index_path) #built in faiss save function. give it index object and file path it does the rest
    print("Pipeline saved.")

def load_pipeline(chunks_path="embedded_chunks.json", index_path="faiss_index.bin"):
    with open(chunks_path, "r") as f:
        embedded_chunks = json.load(f) #read json text with chunk dictionaries
    index = faiss.read_index(index_path) #read index object from binary file
    print("Pipeline loaded from disk.")
    return embedded_chunks, index #returns both read objects

def retrieve_top_chunks_old(query, chunks, top_k=3):
    query_embedding = embed_query(query)
    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True) #new syntax: lambda x: x[0] means take input x and return its first element

    return scored_chunks[:top_k] #returns top 3 most similar chunks

def retrieve_top_chunks(query, embedded_chunks, index, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32") #converts query to 2d numpy array, axis 0 is the query, axis 1 is the embedding. It expects a batch of queries which are stores along axis 0
    #converts to float32 to match the type of the embeddings    
    faiss.normalize_L2(query_embedding) #euclidean norm

    scores, indices = index.search(query_embedding, top_k) #returns the top k most similar chunks stored in index to the query embedding
    #scores.shape = indices.shape = (1,3), so scores[0] returns the 3 similarity scores for the top 3, and indices[0] returns the positions of the best 3 chunks for the query

    top_chunks = [] #empty list to store the top chunks dictionaries

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1): #pairs up the similarity scores and indices for the top chunks relative to the query
        #enumerate also assigns them a 'rank', so the tuple (score, idx) stores the score and index for the top chunks
        chunk = embedded_chunks[idx] #picks out the chunks based on their index in the original list
        top_chunks.append((float(score), chunk)) #appends the chunks to top_chunks, in a tuple with their score
    
    return top_chunks #returns (score, chunk), where chunk is the dictionary including embeddings

def find_references_start(text):
    lines = text.splitlines() #splits into separate lines

    for i, line in enumerate(lines):
        cleaned = line.strip() #removes whitespace from start and end of string, cleaned holds the current stripped line

        if re.fullmatch(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)", cleaned): #checks if cleaned matches any of the options listed
            return i #returns line index
    
    return None

def remove_references_section(pages): #motivated by the fact we dont want top chunks to be a list of references
    cleaned_pages = []

    for page in pages:
        text = page["text"]
        start_idx = find_references_start(text) #tries to find the index of references header

        if start_idx is None: #if it doesnt find the references header
            cleaned_pages.append(page) #append page
        else: #if it does
            lines = text.splitlines() #split text up into lines
            kept_text = "\n".join(lines[:start_idx]) #and keep the text up to the line with the header

            if kept_text.strip(): #strip all whitespace from start and end of kept_text, if this isn't empty, ie there is some text, this is true
                cleaned_pages.append({"page_number": page["page_number"], "text": kept_text}) #if there is text we append the kept_text of the page
                break #end since all the rest is references

    return cleaned_pages

def build_context(top_chunks):
    context = ""

    for i, (chunk, score) in enumerate(top_chunks, start=1):
        context += f"chunk {i} (Page {chunk.metadata['page']}):\n"
        context += chunk.page_content +"\n\n"

    return context

def build_prompt(query, context):
    prompt = f"""Use the following context to answer the question.
Cite the page numbers you draw from in your answer (e.g. Based on page 3...)
If the answer isn't contained within the context, state this clearly.

Context:
{context}

Question:
{query}
"""
    return prompt

def ask_llm(prompt):
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {
                "role": "system", # this part is like a model rule sheet
                "content": "Answer only from the supplied context. If the answer isn't in the context, state this clearly."
            },
            {
                "role": "user", # what we want it to answer (given the rules)
                "content": prompt
            }
        ],
        temperature = 0.2 #temperature controls randomness. keeping it low for more focused answers
    )

    return response.choices[0].message.content

def answer_query(query, vectorstore, top_k=3):
    top_chunks = vectorstore.similarity_search_with_score(query, k=top_k)
    context = build_context(top_chunks)
    prompt = build_prompt(query, context)
    answer = ask_llm(prompt)

    sources = [{"page_number": chunk.metadata["page"], 
        "source": chunk.metadata["source"], "score": round(float(score), 4)} #builds a new dictionary containing page number, source (save path) and score
        for chunk, score in top_chunks]
    
    return answer, sources

pdf_path = "economics_study.pdf"

if os.path.exists("faiss_index"):
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings_model, allow_dangerous_deserialization=True) 
    print("Pipeline loaded from disk")
    
else:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings_model) #both embeds each chunk and builds FAISS index simultaneously
    vectorstore.save_local("faiss_index") #replaces save_pipeline, since vectorstore saves both files info anyway.


#=======================current testing============================================================== 
query = "Is DHL the superior logistics firm?"
answer, sources = answer_query(query, vectorstore, top_k=3)
print(query)
print(answer)
print("Sources: ", sources)

#========================previous testing=============================================================
#TESTING PART 1 (importing pdf splitting into chunks and displaying text)

#print("\nSECOND PAGE NUMBER\n")
#print(chunks[1]["page_number"])
#print("\nSECOND PAGE TEXT\n")
#print(chunks[1]["text"])
#print("\n42ND CHUNK\n")
#print(chunks[42]["text"])
#print("\nTOTAL EXTRACTED PAGES\n")
#print(f"Pages with extracted text: {len(pages)}")
#print(f"Pages with extracted text: {len(set(chunk["page_number"] for chunk in chunks))}") #extracts page number from each chunk, set finds list of unique ones, then find length
#print("\nTOTAL CHUNKS EXTRACTED\n")
#print(f"Number of chunks of text: {len(chunks)}")

#TESTING PART 2 (embedding testing)

#print("\n--- BEFORE EMBEDDINGS ---\n")
#print(f"Number of chunks: {len(chunks)}")
#print(f"First chunk page: {chunks[0]['page_number']}")
#print(f"First chunk text sample:\n{chunks[0]['text'][:300]}")

#chunks = add_embeddings(chunks)

#print("\n--- AFTER EMBEDDINGS ---\n")
#print(f"Embedding length: {len(chunks[0]['embedding'])}") #finds number of dimensions/length of embedding list for 1st chunk
#print(f"First 10 values:\n{chunks[0]['embedding'][:10]}") #displays first 10 dimensions for first chunk
#print(f"Keys in first chunk:\n{chunks[0].keys()}") #Displays what keys there are in chunks using .keys()

#==================testing top chunk selection=============================================================================

#top_chunks = retrieve_top_chunks(query, embedded_chunks, top_k=3)

#for i, (score,chunk) in enumerate(top_chunks, start=1):  #so indexing doesnt start from 0
#    print(f"\nResult {i}") #rank of result in terms of similarity to query
#    print(f"Score: {score}") #top chunks is a tuple of (score, chunk), we print score here
#    print(f"Page number: {chunk['page_number']}") #these 3 print out keys from the dictionary for each chunk
#    print(f"Chunk ID: {chunk['chunk_id']}")
#    print(f"Text: {chunk['text'][:1000]}") #first 500 char of text in chunk
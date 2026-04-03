from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re

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

def chunk_pages(pages_data, chunk_size = 1000):
    chunks = []
    chunk_id = 1

    for page in pages_data:
        page_number = page["page_number"]
        text = page["text"]

        for start in range(0, len(text), chunk_size): #goes over all text in each page in 1000 char steps
            chunk_text = text[start:(start + chunk_size)]

            chunks.append({"page_number": page_number, "chunk_id": chunk_id, "text": chunk_text}) #maintains dictionary structure

            chunk_id += 1 #ensures unique id number

    return chunks

def add_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2") #pretrained embedding model as model

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

def retrieve_top_chunks(query, chunks, top_k=3):
    query_embedding = embed_query(query)
    scored_chunks = []

    for chunk in chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda x: x[0], reverse=True) #new syntax: lambda x: x[0] means take input x and return its first element

    return scored_chunks[:top_k] #returns top 3 most similar chunks

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
        start_idx = find_references_start(text)

        if start_idx is None:
            cleaned_pages.append(page)
        else:
            lines = text.splitlines()
            kept_text = "\n".join(lines[:start_idx])

            if kept_text.strip():
                cleaned_pages.append({"page_number": page["page_number"], "text": kept_text})
                break

    return cleaned_pages



pdf_path = "Transformerv3paper.pdf"
pages = load_pdf_text(pdf_path)
#pages = remove_references_section(pages)
chunks = chunk_pages(pages, 1000)
model = SentenceTransformer("all-MiniLM-L6-v2")
embedded_chunks = add_embeddings(chunks)
query = "What does the paper say about transformer architechture?"

top_chunks = retrieve_top_chunks(query, embedded_chunks, top_k=3)

for i, (score,chunk) in enumerate(top_chunks, start=1):  #so indexing doesnt start from 0
    print(f"\nResult {i}")
    print(f"Score: {score}")
    print(f"Page number: {chunk['page_number']}")
    print(f"Text: {chunk['text'][:1000]}") #first 500 char of text in chunk

#current testing 



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


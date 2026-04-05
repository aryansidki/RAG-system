from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import os
from dotenv import load_dotenv
from openai import OpenAI

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

def chunk_pages(pages_data, chunk_size = 1000, overlap = 150):
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

            chunks.append({"page_number": page_number, "chunk_id": chunk_id, "text": chunk_text}) #maintains dictionary structure

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

    for i, (score, chunk) in enumerate(top_chunks, start=1):
        context += f"chunk {i} (Page {chunk['page_number']}):\n"
        context += chunk["text"] +"\n\n"

    return context

def build_prompt(query, chunks, top_k=3):
    top_chunks = retrieve_top_chunks(query, chunks, top_k=top_k)
    context = build_context(top_chunks)
    print(f"Top chunks page numbers: {top_chunks[0]['page_number']}, {top_chunks[1]['page_number']}, {top_chunks[2]['page_number']}")
    print(f"top chunks scores: {top_chunks[0][0]}, {top_chunks[1][0]}, {top_chunks[2][0]}")
    prompt = f"""Use the following context to answer the question.
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

def answer_query(query, embedded_chunks, top_k=3):
    prompt = build_prompt(query, embedded_chunks, top_k=top_k)
    answer = ask_llm(prompt)
    
    return answer

pdf_path = "19201418_5056_CW1.pdf"
pages = load_pdf_text(pdf_path)
#pages = remove_references_section(pages)
chunks = chunk_pages(pages, 1000)

embedded_chunks = add_embeddings(chunks)

#=======================current testing============================================================== 
query = "What are the practical limitations of a H-bridge DC motor controller?"
answer = answer_query(query, embedded_chunks, top_k=3)
print(query)
print(answer)

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
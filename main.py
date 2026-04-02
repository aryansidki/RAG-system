from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

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

pdf_path = "Transformerv3paper.pdf"
pages = load_pdf_text(pdf_path)
chunks = chunk_pages(pages, 1000)
model = SentenceTransformer("all-MiniLM-L6-v2")

#current testing
print("\n--- BEFORE EMBEDDINGS ---\n")
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk page: {chunks[0]['page_number']}")
print(f"First chunk text sample:\n{chunks[0]['text'][:300]}")

chunks = add_embeddings(chunks)

print("\n--- AFTER EMBEDDINGS ---\n")
print(f"Embedding length: {len(chunks[0]['embedding'])}") #finds number of dimensions/length of embedding list for 1st chunk
print(f"First 10 values:\n{chunks[0]['embedding'][:10]}") #displays first 10 dimensions for first chunk
print(f"Keys in first chunk:\n{chunks[0].keys()}") #Displays what keys there are in chunks using .keys()

#testing part 1 (importing pdf splitting into chunks and displaying text)
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


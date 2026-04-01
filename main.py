from pypdf import PdfReader

def load_pdf_text(file_path):
    reader = PdfReader(file_path) #reader is an object representing the pdf, which we can work with
    all_text = []

    print(f"Number of pages: {len(reader.pages)}") #reader.pages gives access to the pages attribute of reader. allows us to access pages by index

    for i, page in enumerate(reader.pages): #loop through pages, assigning each a temp index 
        text = page.extract_text() 

        if text: #python treats 'truthy' non zero objects as true by default
            all_text.append(text)
        else:
            print(f"Warning: page {i+1} returned no extractable text")
    
    return "\n".join(all_text) #uses newline as separator for each object in all_text

pdf_path = "Transformerv3paper.pdf"
text = load_pdf_text(pdf_path)
print(text)

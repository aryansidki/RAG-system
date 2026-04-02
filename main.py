from pypdf import PdfReader

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

pdf_path = "Transformerv3paper.pdf"
pages = load_pdf_text(pdf_path)
print(pages[1]["page_number"])
print(pages[1]["text"])
print(f"Pages with extracted text: {len(pages)}")


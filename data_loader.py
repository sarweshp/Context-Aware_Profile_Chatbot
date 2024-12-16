import os
import PyPDF2
from docx import Document

class DocumentLoader:
    @staticmethod
    def read_pdf(file_path):
        """
        Reads a PDF file and returns its text content.
        
        Args:
            file_path (str): Path to the PDF file.
        
        Returns:
            str: Extracted text from the PDF.
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return None

    @staticmethod
    def read_docx(file_path):
        """
        Reads a .docx file and returns its text content.
        
        Args:
            file_path (str): Path to the .docx file.
        
        Returns:
            str: Extracted text from the document.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
        try:
            doc = Document(file_path)
        except Exception as e:
            raise Exception(f"Failed to open the document. Error: {e}")
        
        full_text = [para.text for para in doc.paragraphs]
        return '\n'.join(full_text)
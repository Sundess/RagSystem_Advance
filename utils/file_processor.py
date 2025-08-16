import os
import PyPDF2
import docx
from pathlib import Path
import streamlit as st


class FileProcessor:
    """
    Handles reading and processing different file types with cleaning capabilities
    """

    def __init__(self):
        # Create directories if they don't exist
        self.raw_dir = Path("data/raw")
        self.processed_dir = Path("data/processed")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def save_uploaded_file(self, uploaded_file):
        """
        Save uploaded Streamlit file to raw directory
        """
        raw_file_path = self.raw_dir / uploaded_file.name

        with open(raw_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return raw_file_path

    def process_file_with_cleaning(self, file_path, gemini_chat):
        """
        Process file: extract text -> clean with Gemini -> save to processed
        """
        # Step 1: Extract raw text
        raw_content = self.read_file(file_path)

        # Step 2: Clean and filter text using Gemini
        cleaned_content = self._clean_text_with_gemini(
            raw_content, gemini_chat)

        # Step 3: Save cleaned content to processed directory
        file_name = Path(file_path).stem
        processed_file_path = self.processed_dir / f"{file_name}_cleaned.txt"
        self.write_file(processed_file_path, cleaned_content)

        return cleaned_content

    def _clean_text_with_gemini(self, raw_text, gemini_chat):
        """
        Clean and filter text using Gemini with context length handling
        """
        # Split text into chunks if it's too long (Gemini has context limits)
        max_chunk_size = 30000  # Conservative limit for Gemini

        if len(raw_text) <= max_chunk_size:
            return self._clean_single_chunk(raw_text, gemini_chat)

        # Process in chunks for large texts
        chunks = self._split_text_into_chunks(raw_text, max_chunk_size)
        cleaned_chunks = []

        for i, chunk in enumerate(chunks):
            st.info(f"Cleaning text chunk {i+1}/{len(chunks)}...")
            cleaned_chunk = self._clean_single_chunk(chunk, gemini_chat)
            cleaned_chunks.append(cleaned_chunk)

        return "\n\n".join(cleaned_chunks)

    def _clean_single_chunk(self, text_chunk, gemini_chat):
        """
        Clean a single chunk of text using Gemini
        """
        cleaning_prompt = f"""
        You are helping to build a knowledge base. Please clean and filter the following text:

        INSTRUCTIONS:
        1. Remove any irrelevant content (headers, footers, page numbers, etc.)
        2. Fix formatting issues and normalize spacing
        3. Correct obvious typos and grammatical errors
        4. Remove duplicate or redundant information
        5. Organize the content in a clear, logical structure
        6. Keep all important factual information intact
        7. Make the text more readable and coherent
        8. If there are bullet points or lists, format them properly
        9. Remove any advertisements, navigation elements, or metadata
        10. Ensure the text flows naturally for knowledge retrieval

        TEXT TO CLEAN:
        {text_chunk}

        CLEANED TEXT:
        """

        try:
            cleaned_text = gemini_chat.generate_simple_answer(cleaning_prompt)

            # Handle potential response length issues
            if len(cleaned_text) > 50000:  # If response is too long
                # Truncate or ask for shorter version
                shortened_prompt = f"""
                The following text is too long. Please provide a concise, well-organized summary 
                that captures all the key information:
                
                {cleaned_text[:40000]}
                
                Concise summary:
                """
                cleaned_text = gemini_chat.generate_simple_answer(
                    shortened_prompt)

            return cleaned_text

        except Exception as e:
            st.warning(f"⚠️ Text cleaning failed: {e}. Using original text.")
            return text_chunk

    def _split_text_into_chunks(self, text, max_size):
        """
        Split text into chunks while trying to preserve sentence boundaries
        """
        chunks = []
        current_chunk = ""
        sentences = text.split('. ')

        for sentence in sentences:
            if len(current_chunk + sentence) < max_size:
                current_chunk += sentence + '. '
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def read_file(self, file_path):
        """
        Read content from various file formats
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        file_extension = file_path.suffix.lower()

        if file_extension == '.txt':
            return self._read_text_file(file_path)
        elif file_extension == '.pdf':
            return self._read_pdf_file(file_path)
        elif file_extension in ['.docx', '.doc']:
            return self._read_docx_file(file_path)
        elif file_extension == '.md':
            return self._read_text_file(file_path)  # Markdown is plain text
        else:
            # Try to read as plain text
            try:
                return self._read_text_file(file_path)
            except:
                raise ValueError(f"Unsupported file format: {file_extension}")

    def _read_text_file(self, file_path):
        """
        Read plain text file
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()

    def _read_pdf_file(self, file_path):
        """
        Read PDF file
        """
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading PDF: {e}")

    def _read_docx_file(self, file_path):
        """
        Read DOCX file
        """
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {e}")

    def write_file(self, file_path, content):
        """
        Write content to a text file
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)

    def get_file_info(self, file_path):
        """
        Get basic information about a file
        """
        file_path = Path(file_path)

        if not file_path.exists():
            return None

        return {
            'name': file_path.name,
            'size': file_path.stat().st_size,
            'extension': file_path.suffix,
            'modified': file_path.stat().st_mtime
        }

    def list_processed_files(self):
        """
        List all files in the processed directory
        """
        return list(self.processed_dir.glob("*.txt"))

    def list_raw_files(self):
        """
        List all files in the raw directory
        """
        return list(self.raw_dir.glob("*"))

    def clear_all_files(self):
        """
        Clear all files from raw and processed directories
        """
        try:
            # Clear raw directory
            for file_path in self.raw_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()

            # Clear processed directory
            for file_path in self.processed_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()

            return True
        except Exception as e:
            print(f"❌ Error clearing files: {e}")
            return False

    def get_directory_stats(self):
        """
        Get statistics about files in directories
        """
        raw_files = list(self.raw_dir.glob("*"))
        processed_files = list(self.processed_dir.glob("*"))

        return {
            'raw_count': len([f for f in raw_files if f.is_file()]),
            'processed_count': len([f for f in processed_files if f.is_file()]),
            'raw_files': [f.name for f in raw_files if f.is_file()],
            'processed_files': [f.name for f in processed_files if f.is_file()]
        }

    def list_supported_formats(self):
        """
        Return list of supported file formats
        """
        return ['.txt', '.pdf', '.docx', '.doc', '.md']

import requests
from PyPDF2 import PdfReader

class FileLoader:
    def __init__(self, file_path=None, url=None, drive_path=None, txt_path=None):
        self.file_path = file_path
        self.url = url
        self.drive_path = drive_path
        self.txt_path = txt_path
        self.text = ""

    def load_pdf(self, path):
        try:
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return ""

    def load_txt(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"Error loading TXT: {e}")
            return ""

    def load_from_url(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error loading from URL: {e}")
            return ""

    def load_from_drive(self, path):
        """
        تحميل ملف من Google Drive (لو رابط مشاركة مباشر).
        لو عندك API أو مكتبة زي pydrive أو gdown ممكن توسّع هنا.
        """
        try:
            if path.startswith("http"):
                # لو رابط مشاركة مباشر
                response = requests.get(path)
                response.raise_for_status()
                return response.text
            else:
                # لو ملف محلي متزامن مع Google Drive
                return self.load_pdf(path) if path.endswith(".pdf") else self.load_txt(path)
        except Exception as e:
            print(f"Error loading from Drive: {e}")
            return ""

    def load(self, file_type="pdf"):
        """
        file_type: "pdf", "txt", "url", "drive"
        """
        if file_type.lower() == "pdf" and self.file_path:
            self.text = self.load_pdf(self.file_path)
        elif file_type.lower() == "txt" and self.txt_path:
            self.text = self.load_txt(self.txt_path)
        elif file_type.lower() == "url" and self.url:
            self.text = self.load_from_url(self.url)
        elif file_type.lower() == "drive" and self.drive_path:
            self.text = self.load_from_drive(self.drive_path)
        else:
            print("Unsupported file type or missing path.")
            self.text = ""
        return self.text
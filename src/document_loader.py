
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document


def load_pdf(file_path):
    file_path = Path(file_path)

    reader = PdfReader(str(file_path))
    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        if not page_text.strip():
            continue

        document = {
            "id": f"pdf_{file_path.stem}_page_{page_number}",
            "text": page_text,
            "metadata": {
                "source": file_path.name,
                "file_type": "pdf",
                "page_number": page_number
            }
        }

        documents.append(document)

    return documents


def load_docx(file_path):
    file_path = Path(file_path)

    doc = Document(str(file_path))
    extracted_text = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()

        if text:
            extracted_text.append(text)

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip()
                for cell in row.cells
            )

            if row_text.strip():
                extracted_text.append(row_text)

    full_text = "\n".join(extracted_text)

    if not full_text.strip():
        return []

    return [{
        "id": f"docx_{file_path.stem}",
        "text": full_text,
        "metadata": {
            "source": file_path.name,
            "file_type": "docx"
        }
    }]


def load_txt(file_path):
    file_path = Path(file_path)

    with open(file_path, "r", encoding="utf-8") as file:
        extracted_text = file.read()

    if not extracted_text.strip():
        return []

    return [{
        "id": f"txt_{file_path.stem}",
        "text": extracted_text,
        "metadata": {
            "source": file_path.name,
            "file_type": "txt"
        }
    }]


def load_csv(file_path):
    file_path = Path(file_path)

    df = pd.read_csv(file_path)

    if df.empty:
        return []

    extracted_text = df.to_string(index=False)

    return [{
        "id": f"csv_{file_path.stem}",
        "text": extracted_text,
        "metadata": {
            "source": file_path.name,
            "file_type": "csv",
            "total_rows": len(df),
            "total_columns": len(df.columns)
        }
    }]


def load_excel(file_path):
    file_path = Path(file_path)

    workbook = pd.ExcelFile(file_path)
    documents = []

    for sheet_name in workbook.sheet_names:
        df = pd.read_excel(
            workbook,
            sheet_name=sheet_name
        )

        if df.empty:
            continue

        extracted_text = df.to_string(index=False)

        document = {
            "id": f"xlsx_{file_path.stem}_{sheet_name}",
            "text": extracted_text,
            "metadata": {
                "source": file_path.name,
                "file_type": "xlsx",
                "sheet_name": sheet_name,
                "total_rows": len(df),
                "total_columns": len(df.columns)
            }
        }

        documents.append(document)

    return documents


def load_document(file_path):
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    file_extension = file_path.suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
        ".txt": load_txt,
        ".csv": load_csv,
        ".xlsx": load_excel
    }

    if file_extension not in loaders:
        raise ValueError(
            f"Unsupported file format: {file_extension}"
        )

    return loaders[file_extension](file_path)

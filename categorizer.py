import re
import sqlite3
import os
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from pathlib import Path

def load_model(model_path=None):
    """
    Loads the fine-tuned model and tokenizer for sequence classification.

    Args:
        model_path (str): Path to the fine-tuned model directory.

    Returns:
        tokenizer, model: The loaded tokenizer and model.
    """
    if model_path is None:
        # Use the absolute path to the 'model' directory
        model_path = os.path.join(os.path.dirname(__file__), 'model')
    # Convert the path to a POSIX-style path
    model_path = Path(model_path).as_posix()
    
    # Debugging information
    print(f"Attempting to load model from path: {model_path}")
    
    # Check if the path exists locally
    if not os.path.exists(model_path):
        print(f"Directory listing for {os.path.dirname(model_path)}:")
        print(os.listdir(os.path.dirname(model_path)))
        raise ValueError(f"Model path does not exist: {model_path}")
    
    # Check the contents of the model directory
    print(f"Contents of the model directory ({model_path}):")
    print(os.listdir(model_path))
    
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    return tokenizer, model

def categorize_questions(extracted_text, tokenizer, model, label_map):
    """
    Categorizes the extracted text into different types of questions using a natural language model.

    Args:
        extracted_text (str): The extracted text from the images.
        tokenizer: The tokenizer for the model.
        model: The fine-tuned model.
        label_map (dict): Mapping from label indices to category names.

    Returns:
        dict: A dictionary where the keys are the question types and the values are lists of questions of that type.
    """
    # Split the text into questions based on patterns like "1.", "2.", etc.
    sections = re.split(r'\n\d+\.\s', extracted_text)
    sections = [section.strip() for section in sections if section.strip()]

    categorized_questions = {}
    for section in sections:
        # Tokenize the section
        inputs = tokenizer(section, return_tensors='pt', truncation=True, padding=True)
        # Get the model prediction
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        category = label_map[prediction]
        # Add the section to the appropriate category
        categorized_questions.setdefault(category, []).append(section)
    return categorized_questions

def initialize_database(db_path='math_exams.db'):
    """
    Initializes the database schema by creating the necessary tables.

    Args:
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Categories table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE
        )
    ''')

    # Create Questions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_id INTEGER,
            text TEXT,
            FOREIGN KEY (category_id) REFERENCES Categories (id)
        )
    ''')

    # Create Images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER,
            image_path TEXT,
            FOREIGN KEY (question_id) REFERENCES Questions (id)
        )
    ''')

    conn.commit()
    conn.close()

def link_questions_to_images(categorized_questions, db_path='math_exams.db'):
    """
    Links categorized questions to their corresponding images in the database.

    Args:
        categorized_questions (dict): A dictionary of categorized questions.
        db_path (str): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for category, questions in categorized_questions.items():
        # Insert category into Categories table
        cursor.execute('INSERT OR IGNORE INTO Categories (name) VALUES (?)', (category,))
        cursor.execute('SELECT id FROM Categories WHERE name = ?', (category,))
        category_id = cursor.fetchone()[0]

        for question in questions:
            # Insert question into Questions table
            cursor.execute('INSERT INTO Questions (category_id, text) VALUES (?, ?)', (category_id, question))
            question_id = cursor.lastrowid

            # Link question to images (assuming image paths are stored in the question text)
            image_paths = re.findall(r'\[image:(.*?)\]', question)
            for image_path in image_paths:
                cursor.execute('INSERT INTO Images (question_id, image_path) VALUES (?, ?)', (question_id, image_path))

    conn.commit()
    conn.close()

if __name__ == "__main__":
    # Directory containing the extracted text files
    text_files_directory = "output/text"

    # Initialize the database schema
    initialize_database()

    # Load the model and tokenizer
    tokenizer, model = load_model()
    label_map = {
        0: "Least squares method",
        1: "Distance calculation",
        2: "Linear transformation",
        3: "Matrix multiplication",
        4: "Determinant calculation",
        5: "Inverse calculation",
        6: "Eigenvalue calculation",
        7: "Reflection"
    }

    # Iterate over all text files in the directory
    for filename in os.listdir(text_files_directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(text_files_directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                extracted_text = file.read()

            categorized = categorize_questions(extracted_text, tokenizer, model, label_map)
            print(f"Results for {filename}:\n")
            for category, questions in categorized.items():
                print(f"{category}:\n")
                for question in questions:
                    print(f"  - {question}")
                print()
            link_questions_to_images(categorized)
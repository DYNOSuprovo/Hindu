# Hindu

**Hindu** is a Python-based project designed to process and analyze PDF documents using vector embeddings. It leverages modern natural language processing techniques to enable efficient information retrieval and semantic search capabilities.

## Features

- **PDF Processing**: Extracts text content from PDF files located in the `pdfs/` directory.
- **Vector Embedding**: Converts textual data into vector representations for advanced analysis.
- **Semantic Search**: Implements search functionalities to find relevant information based on vector similarities.
- **Modular Design**: Structured with separate scripts for building the vector database and testing, ensuring clarity and maintainability.

## Prerequisites

- Python 3.7 or higher
- Recommended to use a virtual environment to manage dependencies

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/DYNOSuprovo/Hindu.git
   cd Hindu
   ```

2. **Set Up Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Build the Vector Database

Process the PDFs and create a vector database for semantic search.

```bash
python build_vector_db.py
```

### 2. Run Tests

Execute test scripts to validate the functionality of the vector database and search capabilities.

```bash
python test.py
python test2.py
python test3.py
```

## Project Structure

```plaintext
Hindu/
├── pdfs/                 # Directory containing PDF files to be processed
├── build_vector_db.py    # Script to build the vector database from PDFs
├── test.py               # Test script 1
├── test2.py              # Test script 2
├── test3.py              # Test script 3
└── requirements.txt      # Python dependencies
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

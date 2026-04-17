# Retail Q&A Tool

A Streamlit application that answers natural-language questions over a retail T-shirt database using LangChain, Groq LLM, few-shot prompting, and MySQL.

## Features

- Ask business questions in plain English
- Converts questions to SQL using few-shot examples
- Queries MySQL and returns the final answer in the UI
- Uses semantic example selection with Chroma + sentence-transformers

## Project Structure

- `main.py` - Streamlit app entry point
- `langchain_helper.py` - LLM, prompt, embeddings, vectorstore, and SQL chain setup
- `few_shots.py` - Few-shot examples for SQL generation
- `requirements.txt` - Python dependencies
- `.streamlit/config.toml` - Streamlit config
- `secret_key.py` - Groq API key source used by the app

## Prerequisites

- Python 3.10+ (recommended: 3.11 or 3.12)
- MySQL database with expected tables/data
- Git (optional)

## Setup (venv)

### Windows (PowerShell)

```powershell
python -m venv retail_qa
.\retail_qa\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)

```bat
python -m venv retail_qa
retail_qa\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv retail_qa
source retail_qa/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root:

```env
db_user=your_mysql_username
db_password=your_mysql_password
db_host=localhost
db_name=your_database_name
```

Set your Groq API key in `secret_key.py`:

```python
groq_api_key = "your_groq_api_key"
```

## Run the App

```bash
streamlit run main.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## Notes

- The app currently formats the response as a numeric value and displays it.
- The SQL chain is configured to use few-shot prompts and semantic similarity selection.

## Troubleshooting

- If you see many terminal logs from `transformers` or `torch`, restart Streamlit after config/code changes.
- If database connection fails, verify `.env` values and MySQL accessibility.
- If dependency install fails on Windows, ensure your Python version and pip are up to date.

## Security Reminder

- Do not commit real credentials.
- Prefer environment variables or a secret manager for production deployments.

## License

For personal/educational use unless you add a project license file.

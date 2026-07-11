# Text-Summarization-App-using-Hugging-Face (T5)


## Overview

This project is an AI-powered Text Summarization API that generates concise summaries from long text passages using the T5-Small transformer model. The application is built with FastAPI and served using Uvicorn, providing a lightweight and scalable REST API for text summarization tasks.

## Features

* Automatic text summarization
* Transformer-based NLP model (T5-Small)
* REST API built with FastAPI
* Fast inference using pre-trained models
* Easy deployment with Uvicorn
* JSON-based request and response handling

## Tech Stack

* Python
* FastAPI
* Uvicorn
* Hugging Face Transformers
* PyTorch
* T5-Small



## How It Works

1. User sends a text input through the API.
2. The text is preprocessed and tokenized.
3. The T5-Small model generates a summary.
4. The summarized text is returned as a JSON response.


### Clone the Repository

```bash
git clone https://github.com/your-username/text-summarization-api.git
cd text-summarization-api
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Environment

Windows:

```bash
venv\Scripts\activate



```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

The API will be available at:

```bash
http://127.0.0.1:8000
```

Interactive API documentation:

```bash
http://127.0.0.1:8000/docs
```

## API Endpoint

### POST /summarize

Request:

```json
{
  "text": "Artificial Intelligence is transforming industries by automating processes and improving decision-making..."
}
```

Response:

```json
{
  "summary": "AI is transforming industries through automation and improved decision-making."
}
```

## Model Information

* Model: T5-Small
* Architecture: Encoder-Decoder Transformer
* Framework: Hugging Face Transformers
* Task: Abstractive Text Summarization

## Applications

* News article summarization
* Document summarization
* Research paper summaries
* Blog and content generation
* Educational content compression

## Future Improvements

* Fine-tuning on domain-specific datasets
* Batch summarization support
* User authentication
* Docker deployment
* Cloud deployment on AWS or Azure
* Multi-language summarization

## Learning Outcomes

Through this project, I gained hands-on experience with:

* Natural Language Processing (NLP)
* Transformer architectures
* Hugging Face ecosystem
* FastAPI development
* Model deployment
* REST API design

## Author

Nitish Kumar

Aspiring AI Engineer | Machine Learning & NLP Enthusiast

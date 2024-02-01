# internal-docs-ccessibility-POC

## Description

Making easy how developers interact with extensive documentation, addressing the challenge faced by Company X where developers spend significant time navigating or querying peers about AWS documentation. This Proof of Concept (POC) demonstrates the potential to reduce search time by intelligently matching queries with relevant documentation segments, leveraging AWS cloud services and OpenAI's advanced language models.

## Features

- **Automated Documentation Embeddings**: Processes documentation to create semantic embeddings, facilitating efficient query matching.
- **Intelligent Query Response**: Utilizes OpenAI's GPT-3.5 for dynamic, context-aware responses to user queries, pointing towards relevant documentation sections.
- **Cloud Integration**: Fully compatible with AWS services, ensuring scalability and security, especially for handling sensitive internal documentation.

## Installation

Ensure you have Python 3.x and pip installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/davirami88/internal-docs-ccessibility-POC.git
cd internal-docs-ccessibility-POC
pip install -r requirements.txt

```

## Usage

**Flow description** https://excalidraw.com/#room=56260809cd6e764d0519,blpuojSmVyAmm5VN7p1JBA

1. Generating Documentation Embeddings: The embedding files are already generated, but you can run docs_to_embeddings.py to process extra documentation and generate the embeddings.
2. Query Response System: Modify the query on query_response.py to start querying the documentation. Input a query related to AWS documentation (already stored on the .csv files) to receive relevant information and suggestions for further reading.

## Configuration

OpenAI API key for embeddings and query processing.
AWS S3 for storage. (In the next version)

## Acknowledgments

Loka and Company X for the collaboration opportunity.
OpenAI for the language models and embedding technology.

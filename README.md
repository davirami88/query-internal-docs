# Internal Documentation Accessibility (POC)

## Description

Making easy how developers interact with extensive documentation, addressing the challenge faced by Company X where developers spend significant time navigating or querying peers about AWS documentation. This Proof of Concept (POC) demonstrates the potential to reduce search time by intelligently matching queries with relevant documentation segments, leveraging OpenAI's advanced language models.

**Flow description** https://excalidraw.com/#room=56260809cd6e764d0519,blpuojSmVyAmm5VN7p1JBA


## Features

- **Automated Documentation Embeddings**: Processes documentation to create semantic embeddings, facilitating efficient query matching.
- **Intelligent Query Response**: Utilizes OpenAI's GPT-3.5 for dynamic, context-aware responses to user queries, pointing towards relevant documentation sections.
- **Cloud Integration**: Fully compatible with AWS services, ensuring scalability and security, especially for handling sensitive internal documentation. (In the next version)


## Example Queries and Responses

Below are some examples of how the system can respond to various queries related to AWS documentation:

### Query 1
**Q: What is SageMaker?**

**A:** Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models. It offers first-party templates for organizations to quickly get started with ML workflows and CI/CD, using AWS-native services such as AWS CodeBuild, AWS CodePipeline, and AWS CodeCommit. SageMaker Project helps organizations set up and standardize developer environments for data scientists and CI/CD systems for MLOps engineers, providing dependency management, code repository management, build reproducibility, and artifact sharing.

### Query 2
**Q: What are all AWS regions where SageMaker is available?**

**A:** SageMaker is available in the following AWS regions: 'us-east-1', 'us-east-2', 'us-west-1', 'us-west-2', 'ap-northeast-1', 'ap-northeast-2', 'ap-southeast-1', 'ap-southeast-2', 'eu-central-1'.

### Query 3
**Q: How to check if an endpoint is KMS encrypted?**

**A:** To check if an endpoint is KMS encrypted, you can look at the `EndpointConfigWithDataCapture` resource in the CloudFormation template. If the `KmsKeyId` property is specified within the `DataCaptureConfig` section, then the endpoint is KMS encrypted.

### Query 4
**Q: What are SageMaker Geospatial capabilities?**

**A:** SageMaker Geospatial capabilities are a set of managed services that perform operations on behalf of the user on AWS hardware managed by SageMaker. These capabilities can only perform operations that the user permits and require an IAM execution role to grant the service permission to access AWS resources.


## Installation

Ensure you have Python 3.x and pip installed. Clone the repository and install the required dependencies:

```bash
git clone https://github.com/davirami88/internal-docs-POC.git
cd internal-docs-POC
pip install -r requirements.txt

```

## Usage

1. **Generating Documentation Embeddings:** The embedding files are already generated, but you can run docs_to_embeddings.py to process extra documentation and generate the embeddings.
2. **Query Response System:** Modify the query on query_response.py to start querying the documentation. Input a query related to AWS documentation (already stored on the .csv files) to receive relevant information and suggestions for further reading.

## Configuration

OpenAI API key for embeddings and query processing.
AWS S3 for storage. (In the next version)

## Acknowledgments

Loka for the collaboration opportunity.
OpenAI for the language models and embedding technology.

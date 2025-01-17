# Linkage LM Module

## Overview

The Fuzzy LM Module is designed to address the challenge of matching identifiers when they are inconsistent strings that loosely match. This is a common problem in data integration and cleaning tasks where identifiers may vary slightly in spelling, format, or representation. The module provides six robust solution options to tackle this issue:

1. **Fuzzy Multivariate String Matching**: Utilizes the Record-linkage library to compare strings based on similarity metrics and associated column values.
2. **Fuzzy Phonetic String Matching**: Employs phonetic algorithms to match strings that sound similar.
3. **Google Search Completion**: Enhances matching by retrieving additional context and data through web searches.
4. **Vector Embedding Matching**: Leverages vector embeddings to find semantic similarities between identifiers.
5. **Matching with LLMs**: Uses the LLAMA 3 API to perform advanced matching using language models.
6. **Combined Matching**: Integrates multiple matching methods to improve accuracy and reliability.

The module integrates seamlessly into data processing workflows and leverages advanced tools such as the Record-linkage library for data comparison and Embedchain for embedding management. It is designed with usability in mind, enabling easy integration and customization.


---

## Features

- Comprehensive fuzzy matching using multiple methods.
- Management of missing values in datasets.
- Integration with LLAMA 3 API for advanced matching capabilities.
- Embedding operations using Embedchain.
- Flexible configuration and modular design.

---

## File Structure

- **`fuzzy_lm/`**: Core Python package.
  - **`__init__.py`**: Marks the directory as a Python package.
  - **`main.py`**: Entry point for executing the fuzzy matching process.
  - **`fuzzy_matching.py`**: Contains the `FuzzyMatchingAgent` class to orchestrate the operations.
  - **`data_handler.py`**: Handles data loading and saving operations.
  - **`embedchain_integration.py`**: Manages data embeddings.
  - **`llama_api.py`**: Interfaces with the LLAMA 3 API for matching.
  - **`config.py`**: Handles configuration settings.
  - **`docs/documentation.md`**: Detailed documentation for the module.

- **`resources/`**: Contains assets and supporting files for the module.
  - `api_spec_and_task/`: API specifications and task-related files.
  - `code_plan_and_change/`: Code plans and change records.
  - `code_summary/`: Summaries of the module’s codebase.
  - `competitive_analysis/`: Analysis reports.
  - `data_api_design/`: Designs related to the data API.
  - `graph_db/`: Graph database resources.
  - `prd/`: Product requirement documents.
  - `sd_output/`: Outputs of system design processes.
  - `seq_flow/`: Sequence flow diagrams.
  - `system_design/`: System design documents.

- **`tests/`**: Unit tests for the module.
- **`requirements.txt`**: List of required Python packages.

---

## Classes and Key Methods

### FuzzyMatchingAgent
- **`handle_missing_values(dataset: DataFrame) -> DataFrame`**: Handles missing values in datasets.
- **`configure(parameters: dict) -> None`**: Configures the agent with user-defined settings.
- **`process_data(input_file: str, output_file: str) -> None`**: Processes data by embedding, matching, and saving results.

### DataHandler
- **`load_data(file_path: str) -> DataFrame`**: Loads a CSV file into a DataFrame.
- **`save_data(dataset: DataFrame, file_path: str) -> None`**: Saves a DataFrame to a CSV file.

### EmbedchainIntegration
- **`embed_data(dataset: DataFrame) -> list`**: Embeds data from a DataFrame.
- **`retrieve_embeddings(query: str) -> list`**: Retrieves embeddings for a query.

### LLAMAAPI
- **`fuzzy_match(data1: list, data2: list) -> list`**: Performs fuzzy matching between two sets of embeddings.

### Config
- **`load_config(file_path: str) -> dict`**: Loads configuration settings from a JSON file.
- **`save_config(config: dict, file_path: str) -> None`**: Saves configuration settings to a JSON file.

---

## Usage

1. **Prepare Configuration**:
   - Ensure the configuration file (e.g., `config.json`) is properly set up with necessary API credentials and settings.

2. **Run Fuzzy Matching**:
   - Execute the `main()` function in `main.py` to process data and perform matching operations.

3. **Integration**:
   - Use the `FuzzyMatchingAgent` class directly for embedding this functionality in custom workflows.

---

## Dependencies

- Python 3.8+
- **Required Libraries**:
  - `pandas==1.3.3`
  - `recordlinkage==0.14`
  - `embedchain==0.1.0`
  - `llama3api==0.2.1`

Install dependencies with:
```bash
pip install -r requirements.txt
```
---

## Notes

- **LLAMA 3 API**: Ensure API credentials are valid and adhere to usage limits.
- **Customization**: Logic for handling missing values and embeddings is customizable.
- **Placeholders**: Replace placeholder logic for embeddings and matching with production-ready API calls as needed.

---

## Testing

Run unit tests to validate the module's functionality:
```bash
pytest tests/
```

---

## Contact and Support

For issues, questions, or contributions, please refer to the module’s [documentation](fuzzy_lm/docs/documentation.md) or submit an issue in the project repository.

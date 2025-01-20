## docs/documentation.md

# Fuzzy Matching Module Documentation

## Overview

This module provides a comprehensive solution for fuzzy matching and handling missing values using the LLAMA 3 API. It integrates seamlessly with existing data processing pipelines, leveraging the Record-linkage library for data comparison and Embedchain for embedding management. The module is designed to be user-friendly and well-documented to facilitate ease of use and integration.

## File Structure

- **main.py**: Contains the `Main` class and the `main()` function, which serves as the entry point for executing the fuzzy matching process.
- **fuzzy_matching.py**: Implements the `FuzzyMatchingAgent` class, which orchestrates the fuzzy matching operations using various components.
- **data_handler.py**: Provides the `DataHandler` class with methods for loading and saving data.
- **embedchain_integration.py**: Contains the `EmbedchainIntegration` class with methods for embedding data.
- **llama_api.py**: Implements the `LLAMAAPI` class with methods for performing fuzzy matching.
- **config.py**: Provides the `Config` class with methods for loading and saving configuration settings.
- **docs/documentation.md**: This documentation file.

## Class and Method Descriptions

### Main

- **Attributes**:
  - `agent`: An instance of `FuzzyMatchingAgent`.

- **Methods**:
  - `main()`: Executes the data processing workflow.

### FuzzyMatchingAgent

- **Attributes**:
  - `config`: Configuration settings loaded from a file.
  - `data_handler`: An instance of `DataHandler`.
  - `embedchain`: An instance of `EmbedchainIntegration`.
  - `llama_api`: An instance of `LLAMAAPI`.

- **Methods**:
  - `handle_missing_values(dataset: DataFrame) -> DataFrame`: Handles missing values in the dataset.
  - `configure(parameters: dict) -> None`: Configures the agent with given parameters.
  - `process_data(input_file: str, output_file: str) -> None`: Processes data by loading, embedding, matching, and saving.

### DataHandler

- **Methods**:
  - `load_data(file_path: str) -> DataFrame`: Loads data from a CSV file into a DataFrame.
  - `save_data(dataset: DataFrame, file_path: str) -> None`: Saves a DataFrame to a CSV file.

### EmbedchainIntegration

- **Attributes**:
  - `embedding_model`: The name of the embedding model to use.

- **Methods**:
  - `embed_data(dataset: DataFrame) -> list`: Embeds data from a DataFrame.
  - `retrieve_embeddings(query: str) -> list`: Retrieves embeddings for a given query string.

### LLAMAAPI

- **Attributes**:
  - `api_key`: The API key for authentication.
  - `api_url`: The base URL for the LLAMA 3 API.

- **Methods**:
  - `fuzzy_match(data1: list, data2: list) -> list`: Performs fuzzy matching between two sets of data embeddings.

### Config

- **Attributes**:
  - `config`: The current configuration dictionary.

- **Methods**:
  - `load_config(file_path: str) -> dict`: Loads configuration from a JSON file.
  - `save_config(config: dict, file_path: str) -> None`: Saves the configuration to a JSON file.
  - `get_config() -> dict`: Returns the current configuration.
  - `set_config(key: str, value: any) -> None`: Sets a configuration value.

## Usage

1. **Configuration**: Ensure the configuration file (`config.json`) is set up with the necessary API credentials and settings.
2. **Data Processing**: Use the `Main` class to execute the fuzzy matching process by calling the `main()` function.
3. **Integration**: The module can be integrated into existing data processing pipelines by utilizing the `FuzzyMatchingAgent` class.

## Dependencies

- Python packages: `pandas==1.3.3`, `recordlinkage==0.14`, `embedchain==0.1.0`, `llama3api==0.2.1`

## Notes

- Ensure that the LLAMA 3 API credentials are valid and that the API usage limits are adhered to.
- The module is designed to handle missing values by filling them with a placeholder. This behavior can be customized as needed.
- The embedding and fuzzy matching processes are simulated in this implementation. Replace the placeholder logic with actual calls to the respective libraries or APIs for production use.

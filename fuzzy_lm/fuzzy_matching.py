## fuzzy_matching.py

from typing import Dict, List
from data_handler import DataHandler
from embedchain_integration import EmbedchainIntegration
from llama_api import LLAMAAPI
from config import Config
import pandas as pd


class FuzzyMatchingAgent:
    """A class to orchestrate fuzzy matching operations using various components."""

    def __init__(self, config_file: str = "config.json"):
        """Initializes the FuzzyMatchingAgent with necessary components.

        Args:
            config_file (str): The path to the configuration file.
        """
        self.config = Config().load_config(config_file)
        self.data_handler = DataHandler()
        self.embedchain = EmbedchainIntegration(embedding_model=self.config.get("embedding_model", "default_model"))
        self.llama_api = LLAMAAPI(
            api_key=self.config.get("api_key", "default_api_key"),
            api_url=self.config.get("api_url", "https://api.default.com")
        )

    def handle_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to process.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
        # Example strategy: fill missing values with a placeholder
        filled_dataset = dataset.fillna("missing")
        print("Missing values handled in the dataset.")
        return filled_dataset

    def configure(self, parameters: Dict[str, any]) -> None:
        """Configures the agent with given parameters.

        Args:
            parameters (dict): A dictionary of configuration parameters.
        """
        for key, value in parameters.items():
            self.config.set_config(key, value)
        print("Configuration updated with provided parameters.")

    def process_data(self, input_file: str, output_file: str, match_columns: List[str]) -> None:
        """Processes data by loading, selecting columns, embedding, matching, and saving.

        Args:
            input_file (str): The path to the input data file.
            output_file (str): The path to the output data file.
            match_columns (List[str]): List of column names to use for matching.
        """
        # Load data
        dataset = self.data_handler.load_data(input_file)
        print(f"Loaded data from {input_file} with {len(dataset)} records.")

        # Handle missing values
        dataset = self.handle_missing_values(dataset)

        # Validate selected columns
        missing_columns = [col for col in match_columns if col not in dataset.columns]
        if missing_columns:
            raise ValueError(f"The following columns are missing in the input file: {', '.join(missing_columns)}")
        print(f"Selected columns for matching: {', '.join(match_columns)}")

        # Select and combine the specified columns for embedding
        selected_data = dataset[match_columns].astype(str).agg(' '.join, axis=1)
        print("Selected columns have been combined for embedding.")

        # Embed the combined data
        embeddings = self.embedchain.embed_data(selected_data)
        print("Data has been embedded using the embedding model.")

        # Perform fuzzy matching using the LLAMA API
        # Assuming `fuzzy_match` takes embeddings and returns matched data
        matched_data = self.llama_api.fuzzy_match(embeddings, embeddings)
        print("Fuzzy matching has been performed on the embeddings.")

        # Convert matched data to DataFrame
        matched_df = pd.DataFrame(matched_data)
        print(f"Matched data contains {len(matched_df)} records.")

        # Save the matched data to the output file
        self.data_handler.save_data(matched_df, output_file)
        print(f"Matched data has been saved to {output_file}.")

        print("Data processing complete.")

## fuzzy_matching.py

from typing import Dict
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
        self.llama_api = LLAMAAPI(api_key=self.config.get("api_key", "default_api_key"),
                                  api_url=self.config.get("api_url", "https://api.default.com"))

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

    def process_data(self, input_file: str, output_file: str) -> None:
        """Processes data by loading, embedding, matching, and saving.

        Args:
            input_file (str): The path to the input data file.
            output_file (str): The path to the output data file.
        """
        dataset = self.data_handler.load_data(input_file)
        dataset = self.handle_missing_values(dataset)
        embeddings = self.embedchain.embed_data(dataset)
        matched_data = self.llama_api.fuzzy_match(embeddings, embeddings)
        matched_df = pd.DataFrame(matched_data)
        self.data_handler.save_data(matched_df, output_file)
        print("Data processing complete.")

# Example usage
if __name__ == "__main__":
    agent = FuzzyMatchingAgent()
    agent.process_data("input.csv", "output.csv")

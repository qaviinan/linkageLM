## data_handler.py

import pandas as pd
from typing import Optional

class DataHandler:
    """A class to handle data loading and saving operations."""

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file into a DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}.")
            return data
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            return pd.DataFrame()  # Return an empty DataFrame on failure
        except pd.errors.EmptyDataError:
            print(f"No data found in the file {file_path}.")
            return pd.DataFrame()  # Return an empty DataFrame on failure
        except pd.errors.ParserError:
            print(f"Error parsing the file {file_path}.")
            return pd.DataFrame()  # Return an empty DataFrame on failure

    def save_data(self, dataset: pd.DataFrame, file_path: str) -> None:
        """Saves a DataFrame to a CSV file.

        Args:
            dataset (pd.DataFrame): The DataFrame to save.
            file_path (str): The path to the CSV file.
        """
        try:
            dataset.to_csv(file_path, index=False)
            print(f"Data saved successfully to {file_path}.")
        except IOError as e:
            print(f"An error occurred while writing to the file {file_path}: {e}")

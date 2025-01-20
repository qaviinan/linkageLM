## config.py

import json
from typing import Dict

class Config:
    """A class to handle configuration loading and saving."""

    def __init__(self, default_config: Dict[str, any] = None):
        """Initializes the Config class with a default configuration."""
        if default_config is None:
            default_config = {
                "api_key": "default_api_key",
                "api_url": "https://api.default.com",
                "timeout": 30,
                "retry_attempts": 3
            }
        self.config = default_config

    def load_config(self, file_path: str) -> Dict[str, any]:
        """Loads configuration from a JSON file.

        Args:
            file_path (str): The path to the configuration file.

        Returns:
            dict: The configuration dictionary.
        """
        try:
            with open(file_path, 'r') as file:
                self.config = json.load(file)
        except FileNotFoundError:
            print(f"Configuration file {file_path} not found. Using default configuration.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from the configuration file {file_path}. Using default configuration.")
        return self.config

    def save_config(self, config: Dict[str, any], file_path: str) -> None:
        """Saves the configuration to a JSON file.

        Args:
            config (dict): The configuration dictionary to save.
            file_path (str): The path to the configuration file.
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(config, file, indent=4)
        except IOError as e:
            print(f"An error occurred while writing to the file {file_path}: {e}")

    def get_config(self) -> Dict[str, any]:
        """Returns the current configuration.

        Returns:
            dict: The current configuration dictionary.
        """
        return self.config

    def set_config(self, key: str, value: any) -> None:
        """Sets a configuration value.

        Args:
            key (str): The configuration key.
            value (any): The value to set for the key.
        """
        self.config[key] = value

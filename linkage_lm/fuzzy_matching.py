from typing import Dict, List
import pandas as pd
from recordlinkage import Compare, Index
from recordlinkage.preprocessing import phonetic, clean
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from web_reader import researchAgent
from data_handler import DataHandler
from embedchain_integration import EmbedchainIntegration
from llama_api import LLAMAAPI
from config import Config

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
        self.openai_api_key = self.config.get("openai_api_key", "your_openai_api_key_here")

    def handle_missing_values(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Handles missing values in the dataset.

        Args:
            dataset (pd.DataFrame): The dataset to process.

        Returns:
            pd.DataFrame: The dataset with missing values handled.
        """
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

    def fuzzy_string_matching(self, series1: pd.Series, series2: pd.Series, threshold=0.6) -> List:
        """Perform fuzzy string matching between two series using recordlinkage."""
        df1 = pd.DataFrame(series1, columns=['value'])
        df2 = pd.DataFrame(series2, columns=['value'])

        indexer = Index()
        indexer.full()
        candidates = indexer.index(df1, df2)

        compare = Compare()
        compare.string('value', 'value', method='jarowinkler', threshold=threshold, label='string_match')

        features = compare.compute(candidates, df1, df2)

        matches = features[features['string_match'] > threshold].reset_index()
        results = [(series1[i], series2[j], features.loc[(i, j), 'string_match']) for i, j in zip(matches['level_0'], matches['level_1'])]
        return results

    def fuzzy_phonetic_matching(self, series1: pd.Series, series2: pd.Series, threshold=0.75) -> List:
        """Perform fuzzy phonetic matching between two series."""
        series1_phonetic = phonetic(clean(series1), method='soundex')
        series2_phonetic = phonetic(clean(series2), method='soundex')
        matches = []
        for item1, phonetic1 in zip(series1, series1_phonetic):
            for item2, phonetic2 in zip(series2, series2_phonetic):
                if phonetic1 == phonetic2:
                    matches.append((item1, item2, 100))
        return matches

    def google_search_completion(self, series: pd.Series, instructions=None, verbose=True) -> List:
        """Perform Google search completion to retrieve additional data."""
        results = []
        for item in series:
            print(f"Searching for: {item}")
            result = researchAgent(f'{item}', instructions, verbose)
            results.append((item, result))
        return results

    def vector_embedding_matching(self, series1: pd.Series, series2: pd.Series, threshold=0.9) -> List:
        """Match using vector embeddings."""
        def get_embeddings(input_series):
            input_values = list(input_series.values)
            input_keys = input_series.index

            resp = openai.Embedding.create(
                input=input_values,
                engine="text-similarity-davinci-001",
                api_key=self.openai_api_key
            )

            embeds = {}
            for i in range(len(input_values)):
                embeds[input_keys[i]] = resp['data'][i]['embedding']
            return embeds

        series1_embeds = get_embeddings(series1)
        series2_embeds = get_embeddings(series2)

        matches = []
        for key1, embed1 in series1_embeds.items():
            for key2, embed2 in series2_embeds.items():
                similarity = cosine_similarity(np.array(embed1), np.array(embed2))
                if similarity >= threshold:
                    matches.append((key1, key2, similarity))
        return matches

    def llm_matching(self, series1: pd.Series, series2: pd.Series) -> List:
        """Perform fuzzy matching using the LLAMA API."""
        # Assuming `fuzzy_match` takes embeddings and returns matched data
        embeddings1 = self.embedchain.embed_data(series1)
        embeddings2 = self.embedchain.embed_data(series2)
        matched_data = self.llama_api.fuzzy_match(embeddings1, embeddings2)
        return matched_data

    def combined_matching(self, series1: pd.Series, series2: pd.Series) -> List:
        """Combine multiple matching methods."""
        string_matches = self.fuzzy_string_matching(series1, series2)
        phonetic_matches = self.fuzzy_phonetic_matching(series1, series2)
        combined_results = string_matches + phonetic_matches
        return combined_results

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
        matched_data = self.llama_api.fuzzy_match(embeddings, embeddings)
        print("Fuzzy matching has been performed on the embeddings.")

        # Convert matched data to DataFrame
        matched_df = pd.DataFrame(matched_data)
        print(f"Matched data contains {len(matched_df)} records.")

        # Save the matched data to the output file
        self.data_handler.save_data(matched_df, output_file)
        print(f"Matched data has been saved to {output_file}.")

        print("Data processing complete.")

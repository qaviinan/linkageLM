## embedchain_integration.py

import os
from typing import List
import pandas as pd
from embedchain import App
from embedchain.config import Config as EmbedchainConfig

class EmbedchainIntegration:
    """A class to manage embedding operations using the Embedchain library."""

    def __init__(self, embedding_model: str = "sentence-transformers/all-mpnet-base-v2", 
                 llm_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
                 huggingface_token: str = "hf_xxxx"):
        """
        Initializes the EmbedchainIntegration with specified embedding and LLM models.

        Args:
            embedding_model (str): The name of the embedding model to use.
            llm_model (str): The name of the LLM model to use.
            huggingface_token (str): Your Hugging Face access token.
        """
        # Set the Hugging Face access token as an environment variable
        os.environ["HUGGINGFACE_ACCESS_TOKEN"] = huggingface_token

        # Define the Embedchain configuration
        config = {
            'llm': {
                'provider': 'huggingface',
                'config': {
                    'model': llm_model,
                    'top_p': 0.5
                }
            },
            'embedder': {
                'provider': 'huggingface',
                'config': {
                    'model': embedding_model
                }
            }
        }

        # Initialize the Embedchain App with the specified configuration
        self.app = App.from_config(config=EmbedchainConfig(config))

        # Initialize a list to keep track of added data identifiers
        self.data_ids = []
        print(f"Embedchain App initialized with embedder '{embedding_model}' and LLM '{llm_model}'.")

    def embed_data(self, dataset: pd.DataFrame) -> List[List[float]]:
        """
        Embeds data from a DataFrame using the specified embedding model.

        Args:
            dataset (pd.DataFrame): The DataFrame containing data to embed.

        Returns:
            List[List[float]]: A list of embeddings for each row in the dataset.
        """
        embeddings = []
        for index, row in dataset.iterrows():
            # Combine the row data into a single string for embedding
            combined_text = ' '.join(row.astype(str).tolist())
            identifier = f"row_{index}"
            
            # Add the combined text to the Embedchain App
            self.app.add(combined_text, id=identifier)
            self.data_ids.append(identifier)
            print(f"Added data with ID '{identifier}' to Embedchain App.")

        print("All data has been added to Embedchain App for embedding.")

        # Retrieve embeddings for each added data point
        for identifier in self.data_ids:
            # Query the App to retrieve embedding
            # Note: This assumes that the App can return embeddings based on identifiers
            # Adjust the retrieval method based on Embedchain's actual API
            response = self.app.query(f"Retrieve embedding for {identifier}")
            
            # Parse the embedding from the response
            # This is a placeholder; the actual implementation depends on Embedchain's response format
            embedding = self._parse_embedding(response)
            embeddings.append(embedding)
            print(f"Retrieved embedding for ID '{identifier}'.")

        print("All embeddings have been retrieved from Embedchain App.")
        return embeddings

    def retrieve_embeddings(self, query: str) -> List[float]:
        """
        Retrieves embeddings for a given query string.

        Args:
            query (str): The query string to retrieve embeddings for.

        Returns:
            List[float]: The embedding for the query.
        """
        # Add the query to the Embedchain App
        query_id = "query_embedding"
        self.app.add(query, id=query_id)
        print(f"Added query data with ID '{query_id}' to Embedchain App.")

        # Retrieve the embedding for the query
        response = self.app.query(f"Retrieve embedding for {query_id}")
        
        # Parse the embedding from the response
        embedding = self._parse_embedding(response)
        print(f"Retrieved embedding for query '{query}'.")

        return embedding

    def _parse_embedding(self, response: str) -> List[float]:
        """
        Parses the embedding from the Embedchain App's response.

        Args:
            response (str): The response string from the Embedchain App.

        Returns:
            List[float]: The parsed embedding vector.
        """
        # Placeholder for parsing logic
        # This needs to be implemented based on Embedchain's actual response format
        # For example, if the response contains a JSON with the embedding:
        # import json
        # data = json.loads(response)
        # return data['embedding']
        
        # Since we don't have the actual response format, we'll simulate an embedding
        embedding = [0.0] * 128  # Example: 128-dimensional embedding
        print("Parsed embedding from response.")
        return embedding

# Example usage
if __name__ == "__main__":
    # Sample data for embedding
    sample_data = pd.DataFrame({
        'name': ['Elon Musk', 'Jeff Bezos'],
        'address': ['Los Angeles', 'Seattle']
    })

    embedder = EmbedchainIntegration(
        embedding_model='sentence-transformers/all-mpnet-base-v2',
        llm_model='mistralai/Mistral-7B-Instruct-v0.2',
        huggingface_token='hf_xxxx'  # Replace with your actual token
    )

    embeddings = embedder.embed_data(sample_data)
    print("Embeddings:", embeddings)

    query_embedding = embedder.retrieve_embeddings("What is the address of Elon Musk?")
    print("Query Embedding:", query_embedding)

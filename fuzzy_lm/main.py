## main.py

import argparse
from fuzzy_matching import FuzzyMatchingAgent

class Main:
    """Main class to execute the fuzzy matching process."""

    def __init__(self, input_file: str, output_file: str, match_columns: list):
        """
        Initializes the Main class with a FuzzyMatchingAgent.

        Args:
            input_file (str): Path to the input CSV file.
            output_file (str): Path to the output CSV file.
            match_columns (list): List of column names to use for matching.
        """
        self.agent = FuzzyMatchingAgent()
        self.input_file = input_file
        self.output_file = output_file
        self.match_columns = match_columns

    def main(self) -> None:
        """Main function to execute the data processing workflow."""
        self.agent.process_data(self.input_file, self.output_file, self.match_columns)
        print("Fuzzy matching process completed successfully.")

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Perform fuzzy matching on a CSV file.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        default="input.csv",
        help="Path to the input CSV file (default: input.csv)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.csv",
        help="Path to the output CSV file (default: output.csv)"
    )
    parser.add_argument(
        "-c", "--columns",
        type=str,
        nargs='+',
        required=True,
        help="Column names to use for matching (e.g., -c name address)"
    )
    return parser.parse_args()

# Entry point for the script
if __name__ == "__main__":
    args = parse_arguments()
    main_instance = Main(args.input, args.output, args.columns)
    main_instance.main()

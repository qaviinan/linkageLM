## main.py

from fuzzy_matching import FuzzyMatchingAgent

class Main:
    """Main class to execute the fuzzy matching process."""

    def __init__(self):
        """Initializes the Main class with a FuzzyMatchingAgent."""
        self.agent = FuzzyMatchingAgent()

    def main(self) -> None:
        """Main function to execute the data processing workflow."""
        input_file = "input.csv"
        output_file = "output.csv"
        self.agent.process_data(input_file, output_file)
        print("Fuzzy matching process completed successfully.")

# Entry point for the script
if __name__ == "__main__":
    main_instance = Main()
    main_instance.main()

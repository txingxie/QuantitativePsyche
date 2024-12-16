import pandas as pd

def parse_results(input_file, output_file):
    """
    Validates and reformats the results CSV to ensure consistency.

    Args:
        input_file (str): Path to the CSV file containing structured results.
        output_file (str): Path to save the validated results.

    Returns:
        None
    """
    # Load the results from the inference step
    data = pd.read_csv(input_file)

    # Check for required columns
    required_columns = [
        "report", "mood_score", "mood_justification",
        "anxiety_score", "anxiety_justification",
        "depression_score", "depression_justification",
        "suicidality_score", "suicidality_justification"
    ]

    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns in input file: {missing_columns}")

    # Clean up data (remove leading/trailing spaces)
    for col in required_columns:
        data[col] = data[col].astype(str).str.strip()

    # Save the validated and cleaned data
    data.to_csv(output_file, index=False)
    print(f"Validated results saved to {output_file}.")

if __name__ == "__main__":
    input_file = "results.csv"         # File containing structured results
    output_file = "parsed_results.csv" # File to save validated results

    parse_results(input_file, output_file)

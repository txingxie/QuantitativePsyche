import pandas as pd
import os

def read_input_file(file_path):
    """
    Reads input psychiatric records from a CSV or TXT file.

    Args:
        file_path (str): Path to the input file.

    Returns:
        pd.DataFrame: DataFrame containing the records.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # Treat the entire file content as a single report
        data = pd.DataFrame({'report': [content.strip()]})
    else:
        raise ValueError("Unsupported file format. Use a .csv or .txt file.")

    return data

def clean_text(text):
    """
    Cleans and standardizes psychiatric notes.

    Args:
        text (str): Raw text from the record.

    Returns:
        str: Cleaned text.
    """
    # Remove newlines and extra whitespace
    clean_text = text.replace('\n', ' ').strip()
    return clean_text

def format_prompt(report):
    """
    Formats the psychiatric note into a prompt compatible with the Cerebras API.

    Args:
        report (str): Cleaned psychiatric note.

    Returns:
        str: Formatted prompt.
    """
    prompt_template = (
        "[INST] <<SYS>>\n"
        "You are an attentive, specialized psychiatric assistant. Below is a psychiatric history. "
        "Evaluate the following mental health dimensions: mood, anxiety, depression, and suicidality. "
        "For each dimension, assign a severity score (0-10) and provide a detailed justification.\n\n"
        "Output your response in the following JSON format:\n"
        "{{\n"
        "  \"mood\": {{\"score\": X, \"justification\": \"...\"}},\n"
        "  \"anxiety\": {{\"score\": X, \"justification\": \"...\"}},\n"
        "  \"depression\": {{\"score\": X, \"justification\": \"...\"}},\n"
        "  \"suicidality\": {{\"score\": X, \"justification\": \"...\"}}\n"
        "}}\n\n"
        "Replace X with the severity score and provide the justification as a string.\n"
        "<</SYS>>[/INST]\n"
        "History: {report}\n"
        "Question: What are the severity scores for mood, anxiety, depression, and suicidality? "
        "Justify each score in JSON format, providing a detailed explanation.\n"
        "[/INST]"
    )
    return prompt_template.format(report=report)


def preprocess_data(file_path, output_path):
    """
    Reads input data, cleans text, and formats it into prompts.

    Args:
        file_path (str): Path to the input file.
        output_path (str): Path to save the formatted prompts as a CSV.

    Returns:
        None
    """
    # Read input file
    data = read_input_file(file_path)

    # Ensure 'report' column exists
    if 'report' not in data.columns:
        raise KeyError("The input data must contain a 'report' column.")

    # Clean and format prompts
    data['cleaned_report'] = data['report'].apply(clean_text)
    data['prompt'] = data['cleaned_report'].apply(format_prompt)

    # Save to output file
    data[['report', 'prompt']].to_csv(output_path, index=False)
    print(f"Formatted prompts saved to {output_path}.")

if __name__ == "__main__":
    input_file = "input_notes.csv"  # Replace with path to input file
    output_file = "formatted_prompts.csv"  # Replace with desired output path

    preprocess_data(input_file, output_file)

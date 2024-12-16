import pandas as pd
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv
import os
import json

# Load environment variables
load_dotenv()

def initialize_cerebras_client(api_key):
    """
    Initializes the Cerebras client.

    Args:
        api_key (str): API key for Cerebras.

    Returns:
        Cerebras: An instance of the Cerebras client.
    """
    return Cerebras(api_key=api_key)

def send_prompt_to_cerebras(client, prompt, model="llama3.1-8b"):
    """
    Sends a single prompt to the Cerebras API and retrieves the response.

    Args:
        client (Cerebras): The initialized Cerebras client.
        prompt (str): The formatted prompt to send.
        model (str): The Llama model to use (default: llama3.1-8b).

    Returns:
        str: The API response content as a string.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    # Access the content of the first choice
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError) as e:
        print(f"Error accessing response content: {e}")
        print(f"Full response: {response}")
        raise

def parse_response(response):
    """
    Parses the API response (JSON) to extract scores and justifications.

    Args:
        response (str): The raw response from the API (expected in JSON format).

    Returns:
        dict: A dictionary containing scores and justifications for each dimension.
    """
    try:
        parsed_response = json.loads(response)
        return {
            "mood": parsed_response.get("mood", {}),
            "anxiety": parsed_response.get("anxiety", {}),
            "depression": parsed_response.get("depression", {}),
            "suicidality": parsed_response.get("suicidality", {})
        }
    except json.JSONDecodeError as e:
        print(f"Error decoding response JSON: {e}")
        return {
            "mood": {"score": None, "justification": "Invalid response format"},
            "anxiety": {"score": None, "justification": "Invalid response format"},
            "depression": {"score": None, "justification": "Invalid response format"},
            "suicidality": {"score": None, "justification": "Invalid response format"}
        }


def run_inference(input_file, output_file, api_key):
    """
    Processes all prompts in the input file by sending them to the Cerebras API.

    Args:
        input_file (str): Path to the CSV file containing prompts.
        output_file (str): Path to save the structured results.
        api_key (str): API key for Cerebras.

    Returns:
        None
    """
    # Initialize Cerebras
    client = initialize_cerebras_client(api_key)

    # Read the prompts from the CSV
    data = pd.read_csv(input_file)

    results = []
    for _, row in data.iterrows():
        prompt = row["prompt"]
        response = send_prompt_to_cerebras(client, prompt)
        parsed_results = parse_response(response)

        # Include all scores and justifications in the results
        row_result = {
            "report": row["report"],
            "mood_score": parsed_results["mood"]["score"],
            "mood_justification": parsed_results["mood"]["justification"],
            "anxiety_score": parsed_results["anxiety"]["score"],
            "anxiety_justification": parsed_results["anxiety"]["justification"],
            "depression_score": parsed_results["depression"]["score"],
            "depression_justification": parsed_results["depression"]["justification"],
            "suicidality_score": parsed_results["suicidality"]["score"],
            "suicidality_justification": parsed_results["suicidality"]["justification"]
        }

        results.append(row_result)

    # Save the results to a CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")


if __name__ == "__main__":
    input_file = "formatted_prompts.csv"  # Input file with prompts
    output_file = "results.csv"           # File to save inference results

    # Ensure API key is set
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is not set in the environment variables.")

    # Run the inference process
    run_inference(input_file, output_file, api_key)


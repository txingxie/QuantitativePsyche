import os
from dotenv import load_dotenv
from preprocessing import preprocess_data
from inference_pipeline import run_inference
from postprocessing import parse_results

def main():
    """
    Main function to execute the end-to-end pipeline for mental health evaluation.
    """
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        raise ValueError("CEREBRAS_API_KEY is not set in the environment variables.")

    # File paths
    input_file = "admission_notes/patient_notes_1.txt"          # Raw psychiatric notes
    preprocessed_file = "output/formatted_prompts.csv"  # Preprocessed prompts
    inference_output_file = "output/results.csv"  # Raw responses from the inference step
    final_output_file = "output/parsed_results.csv"  # Final parsed results

    # Step 1: Preprocess data
    print("Preprocessing data...")
    preprocess_data(input_file, preprocessed_file)

    # Step 2: Run inference
    print("Running inference...")
    run_inference(preprocessed_file, inference_output_file, api_key)

    # Step 3: Postprocess results
    print("Postprocessing results...")
    parse_results(inference_output_file, final_output_file)

    print("Pipeline completed successfully. Results saved to:", final_output_file)

if __name__ == "__main__":
    main()

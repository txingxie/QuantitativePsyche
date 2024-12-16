# Psychiatric Note Analysis w/ LLMs

## Context

Mental health assessments often rely on detailed clinical notes, which can be time-intensive and laborious to create for psychiatrists aiming to provide the best care possible for their patients. By automating this process, we can:

1. Ensure more consistent and objective mental health assessments.
2. Reduce clinician workload, allowing them to spend more time providing direct care.
3. Provide structured insights for further analysis and research.

## Overview

QuantitativePsyche evaluates psychiatric admission notes to determine severity scores and justifications for mental health dimensions, including:

- **Mood**
- **Anxiety**
- **Depression**
- **Suicidality**

This pipeline parses psychiatric notes, runs inference using LLaMA, and outputs results with severity scores and justifications. By leveraging the speed of **Cerebras Inference**, the processing of enormous volumes of medical text is feasible.

## Usage

### `preprocessing.py`

- Reads raw psychiatric admission notes and formats them into prompts compatible with LLaMA.
- **Output**: `formatted_prompts.csv` — Contains:
  - Original `report`
  - Formatted `prompt` for LLaMA.

### `inference_pipeline.py`

- Directs the formatted prompts to the Cerebras Cloud API and retrieves responses using the LLaMA model.
- **Output**: `results.csv` — Contains:
  - Original `report`
  - Raw responses from the LLaMA model.

### `postprocessing.py`

- Parses and extracts severity scores and justifications for mood, anxiety, depression, and suicidality from the raw LLaMA response.
- **Output**: `parsed_results.csv` — Contains:
  - Original `report`
  - Scores and justifications for each dimension:
    - `mood_score`, `mood_justification`
    - `anxiety_score`, `anxiety_justification`
    - `depression_score`, `depression_justification`
    - `suicidality_score`, `suicidality_justification`

## How to Run the Pipeline

### 1. **Setup Environment**

- Install dependencies:
  ```bash
  pip install pandas cerebras-cloud-sdk python-dotenv
  ```

### 2. **Add Inputs**

- Place input files into the `admission_notes` folder.

### 3. **Execute Inference**

- Run the pipeline:
  ```bash
  python src/main.py
  ```

### 4. **Results**

- Structured results are saved in the output folder:
  - `formatted_prompts.csv` - prompts sent to LLaMa for each patient
  - `results.csv` - raw scores and justifications received from query
  - `parsed_results.csv` - final data output after post-processing

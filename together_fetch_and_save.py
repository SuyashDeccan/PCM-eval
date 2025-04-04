import pandas as pd
from together import Together, error as together_error
import dotenv
import os
import logging
import time
import re  # Import regex module
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm  # Auto-detects notebook/console environment

# --- Configuration ---
# Load environment variables (.env file should contain TOGETHER_API_KEY)
dotenv.load_dotenv()

# Constants
INPUT_CSV = "pending.csv"
OUTPUT_CSV = "together_deepseek_verify.csv"  # Updated output name
MAX_ROWS = 2  # Set to an integer (e.g., 10) to limit rows for testing, None for all
QUESTION_COL = "User Query"  # Make sure this matches your CSV
GROUND_TRUTH_COL = "User Response"  # Make sure this matches your CSV
MODELS_TO_TEST = [
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen2.5-72B-Instruct-Turbo",
    "Qwen/QwQ-32B",
]
VERIFICATION_MODEL = "deepseek-ai/DeepSeek-R1"  # Using DeepSeek R1 for verification
MAX_WORKERS = 8  # Adjusted default, tune based on performance and rate limits, DeepSeek might have different limits
LOG_FILE = "processing_deepseek_verify.log"  # Updated log file name
LOG_LEVEL = logging.INFO  # Change to logging.DEBUG for more detailed logs

# --- Logging Setup ---
# Remove existing handlers if re-running in the same session (e.g., notebook)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(threadName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),  # Also print logs to console
    ],
)
logging.info("--- Script Started ---")

# --- API Client Initialization ---
try:
    if not os.getenv("TOGETHER_API_KEY"):
        raise ValueError("TOGETHER_API_KEY not found in environment variables.")
    client = Together()
    logging.info(f"Successfully initialized Together AI client.")
except ValueError as e:
    logging.error(f"Environment variable error: {e}")
    exit(1)
# except together_error.AuthenticationError: # Uncomment if needed, depends on specific exceptions Together throws
#     logging.error("Authentication Error: Invalid Together API Key.")
#     exit(1)
except Exception as e:
    logging.error(f"Failed to initialize Together client: {e}")
    exit(1)


# --- Prompt Generation Functions ---
def generate_prompt(question):
    prompt_template = """
    Please return the full answer to the following question with a detailed explanation:
    Question: {question}
    """
    return prompt_template.format(question=question)


def generate_verify_prompt(response: str, ground_truth: str) -> str:
    # Updated prompt slightly for clarity with the DeepSeek format expectation
    prompt_template = """
    Verify if the final answer derived from the Response matches the Ground Truth Answer.
    Consider potential variations in phrasing or calculation steps, but focus on the final numerical result or core conclusion.
    Think step-by-step within <think></think> tags.
    Then, provide your final verification answer ONLY as YES or NO after the </think> tag.

    Response:
    {response}

    Ground Truth Answer:
    {ground_truth}

    Verification (Use <think> tags then YES or NO):"""
    return prompt_template.format(response=response, ground_truth=ground_truth)


# --- Core API Interaction Functions (with error handling & logging) ---
def get_answer(question: str, model: str, row_index: int = -1) -> str:
    """Gets an answer from the specified model."""
    logging.debug(f"Requesting answer for row {row_index} from model {model}")
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful math assistant providing detailed step-by-step solutions.",
                },
                {"role": "user", "content": generate_prompt(question)},
            ],
            temperature=0.5,  # Adjust as needed
            max_tokens=1500,  # Adjust as needed
        )
        result = response.choices[0].message.content
        duration = time.time() - start_time
        logging.debug(
            f"Received answer for row {row_index} from {model} in {duration:.2f}s. Length: {len(result)}"
        )
        return result
    except together_error.RateLimitError:
        logging.warning(
            f"Rate limit hit for answer model {model} on row {row_index}. Reduce MAX_WORKERS or wait."
        )
        return "API_ERROR: Rate limit"
    except together_error.APIError as e:
        logging.error(f"API Error for answer model {model} on row {row_index}: {e}")
        return f"API_ERROR: {e}"
    except Exception as e:
        logging.error(
            f"Unexpected error getting answer for model {model} on row {row_index}: {e}",
            exc_info=LOG_LEVEL <= logging.DEBUG,
        )
        return f"ERROR: {e}"


def parse_deepseek_verification(raw_response: str, row_index: int) -> str:
    """Extracts the final YES/NO response after </think> tag."""
    # Regex to find content after the last </think> tag, handling potential whitespace
    # Uses re.DOTALL to make '.' match newlines as well
    match = re.search(r"</think>(.*)", raw_response, re.DOTALL | re.IGNORECASE)

    if match:
        extracted_text = match.group(1).strip()
        logging.debug(
            f"Row {row_index}: Extracted verification text: '{extracted_text}'"
        )
        # Further clean up just in case (take first word if multiple)
        final_word = extracted_text.split()[0] if extracted_text else ""
        return final_word.upper()
    else:
        # If </think> tag is missing, try to find YES or NO directly as a fallback
        logging.warning(
            f"Row {row_index}: </think> tag not found in verification response: '{raw_response[:100]}...' Attempting direct YES/NO extraction."
        )
        # Look for YES or NO as whole words, ignoring case, prioritizing the first match
        direct_match = re.search(r"\b(YES|NO)\b", raw_response, re.IGNORECASE)
        if direct_match:
            extracted_text = direct_match.group(1).strip().upper()
            logging.debug(f"Row {row_index}: Found direct YES/NO: '{extracted_text}'")
            return extracted_text
        else:
            logging.error(
                f"Row {row_index}: Could not extract YES/NO from verification response. Raw: '{raw_response[:100]}...'"
            )
            return "PARSE_ERROR"  # Specific error for parsing failure


def get_verification(
    response_to_verify: str, ground_truth: str, model: str, row_index: int = -1
) -> str:
    """Verifies the response using DeepSeek-R1 and parses its specific output format."""
    # Don't try to verify if the initial answer generation failed
    if not isinstance(response_to_verify, str) or response_to_verify.startswith(
        ("API_ERROR", "ERROR:")
    ):
        logging.warning(
            f"Skipping verification for row {row_index} because answer generation failed: {response_to_verify}"
        )
        return "N/A - Answer Failed"

    if not response_to_verify or not ground_truth:
        logging.warning(
            f"Skipping verification for row {row_index} due to empty response or ground truth."
        )
        return "N/A - Empty Input"

    logging.debug(f"Requesting verification for row {row_index} from model {model}")
    start_time = time.time()
    try:
        verify_response = client.chat.completions.create(
            model=model,  # Should be DeepSeek-R1 as per config
            messages=[
                {
                    "role": "system",
                    "content": "You are a verification assistant. Compare the Response to the Ground Truth Answer. Focus on the final numerical result or core conclusion. Use <think> tags for your reasoning process, then output ONLY YES or NO after the </think> tag.",
                },
                {
                    "role": "user",
                    "content": generate_verify_prompt(response_to_verify, ground_truth),
                },
            ],
            temperature=0.1,  # Low temperature for deterministic YES/NO
            max_tokens=512,  # Allow more tokens for the <think> part
        )
        raw_result = verify_response.choices[0].message.content
        duration = time.time() - start_time
        logging.debug(
            f"Received raw verification for row {row_index} from {model} in {duration:.2f}s."
        )

        # --- Parse the DeepSeek R1 specific format ---
        parsed_result = parse_deepseek_verification(raw_result, row_index)

        # --- Validate the PARSED result ---
        if parsed_result in ["YES", "NO"]:
            return parsed_result  # Return the clean YES or NO
        else:
            # This includes PARSE_ERROR or any other unexpected extracted string
            logging.warning(
                f"Verification model {model} for row {row_index} did not result in YES/NO after parsing. Parsed: '{parsed_result}', Raw: '{raw_result[:100]}...'"
            )
            return f"INVALID_VERIFICATION ({parsed_result})"  # Include parsed result for context

    except together_error.RateLimitError:
        logging.warning(
            f"Rate limit hit for verification model {model} on row {row_index}."
        )
        return "API_ERROR: Rate limit"
    except together_error.APIError as e:
        logging.error(
            f"API Error during verification for model {model} on row {row_index}: {e}"
        )
        return f"API_ERROR: {e}"
    except Exception as e:
        logging.error(
            f"Unexpected error during verification for model {model} on row {row_index}: {e}",
            exc_info=LOG_LEVEL <= logging.DEBUG,
        )
        return f"ERROR: {e}"


# --- Main Execution ---
if __name__ == "__main__":
    logging.info(f"Reading input CSV: {INPUT_CSV}")
    try:
        df = pd.read_csv(INPUT_CSV)
        if MAX_ROWS:
            df = df.head(MAX_ROWS)
            logging.info(f"Processing first {MAX_ROWS} rows.")
        else:
            logging.info(f"Processing all {len(df)} rows.")
    except FileNotFoundError:
        logging.error(f"Input file not found: {INPUT_CSV}")
        exit(1)
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        exit(1)

    # Validate required columns
    if QUESTION_COL not in df.columns:
        logging.error(f"Missing required column in CSV: '{QUESTION_COL}'")
        exit(1)
    if GROUND_TRUTH_COL not in df.columns:
        logging.error(f"Missing required column in CSV: '{GROUND_TRUTH_COL}'")
        exit(1)

    logging.info(f"Input Data Shape: {df.shape}")
    logging.info(f"Found columns: {df.columns.tolist()}")
    logging.info(f"Models to test: {MODELS_TO_TEST}")
    logging.info(f"Verification model: {VERIFICATION_MODEL}")
    logging.info(f"Max concurrent workers: {MAX_WORKERS}")

    total_start_time = time.time()

    # Process each model
    for model in MODELS_TO_TEST:
        model_start_time = time.time()
        logging.info(f"--- Processing Model: {model} ---")
        response_col = model + " response"
        verify_col = model + " response correctness"

        # --- Parallel Answer Generation ---
        logging.info(f"Generating answers using {model}...")
        answers = [None] * len(df)  # Pre-allocate list for correct ordering
        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix=f"{model.split('/')[-1][:10]}_Answer",
        ) as executor:
            future_to_index = {
                executor.submit(get_answer, row[QUESTION_COL], model, index): index
                for index, row in df.iterrows()
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(df),
                desc=f"Answers ({model.split('/')[-1]})",
                unit="row",
            ):
                index = future_to_index[future]
                try:
                    answers[index] = future.result()
                except Exception as e:
                    logging.error(
                        f"Task execution error for answer at index {index}, model {model}: {e}",
                        exc_info=LOG_LEVEL <= logging.DEBUG,
                    )
                    answers[index] = f"TASK_ERROR: {e}"
        df[response_col] = answers
        logging.info(f"Finished generating answers for {model}.")

        # --- Parallel Verification (using VERIFICATION_MODEL) ---
        logging.info(f"Generating verifications using {VERIFICATION_MODEL}...")
        verifications = [None] * len(df)  # Pre-allocate list
        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS,
            thread_name_prefix=f"Verify_{VERIFICATION_MODEL.split('/')[-1][:10]}",
        ) as executor:
            future_to_index = {
                # Pass the newly generated response column for this model
                executor.submit(
                    get_verification,
                    row[response_col],
                    row[GROUND_TRUTH_COL],
                    VERIFICATION_MODEL,
                    index,
                ): index
                for index, row in df.iterrows()
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(df),
                desc=f"Verify ({VERIFICATION_MODEL.split('/')[-1]})",
                unit="row",
            ):
                index = future_to_index[future]
                try:
                    verifications[index] = future.result()
                except Exception as e:
                    logging.error(
                        f"Task execution error for verification at index {index}, model {VERIFICATION_MODEL}: {e}",
                        exc_info=LOG_LEVEL <= logging.DEBUG,
                    )
                    verifications[index] = f"TASK_ERROR: {e}"
        df[verify_col] = verifications
        logging.info(f"Finished generating verifications for {model} responses.")

        model_duration = time.time() - model_start_time
        logging.info(
            f"--- Finished processing Model: {model} in {model_duration:.2f}s ---"
        )
        # Save intermediate results after each model (optional, good for long runs)
        # intermediate_filename = f"testing_intermediate_{model.replace('/', '_')}.csv"
        # logging.info(f"Saving intermediate results to {intermediate_filename}")
        # df.to_csv(intermediate_filename, index=False)

    # --- Save Final Results ---
    logging.info(f"Saving final results to {OUTPUT_CSV}")
    try:
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info("Successfully saved final results.")
    except Exception as e:
        logging.error(f"Failed to save final results to CSV: {e}")

    total_duration = time.time() - total_start_time
    logging.info(f"--- Script Finished in {total_duration:.2f}s ---")

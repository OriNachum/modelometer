# model_comparison.py

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

# ----------------------------
# Configuration Management
# ----------------------------

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        logging.error(f"Failed to load configuration file: {e}")
        sys.exit(1)

# ----------------------------
# Logging Setup
# ----------------------------

def setup_logging(log_level, log_format, log_file=None):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=numeric_level,
                        format=log_format,
                        handlers=handlers)

# ----------------------------
# API Interaction with Streaming
# ----------------------------

def send_streaming_request(api_endpoint, headers, payload):
    try:
        with requests.post(api_endpoint, headers=headers, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                yield line
    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None

# ----------------------------
# Token Counting (Simplistic)
# ----------------------------

def count_tokens(text):
    return len(text.split())

# ----------------------------
# Main Processing Function
# ----------------------------

# Main Processing Function with Defensive Coding and Event Filtering
def process_inputs(config):
    input_path = Path(config['files']['input_csv'])
    output_path = Path(config['files']['output_csv'])

    if not input_path.exists():
        logging.error(f"Input CSV file does not exist: {input_path}")
        sys.exit(1)

    # Prepare output CSV: Write header if file does not exist
    if not output_path.exists():
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'timestamp',
                'model_name',
                'system_prompt',
                'question',
                'response',
                'processing_time_sec',
                'time_to_first_token',
                'tokens_used',
                'tokens_per_second'
            ])

    # Read input CSV using pandas for efficiency
    try:
        df = pd.read_csv(input_path, skipinitialspace=True)
    except Exception as e:
        logging.error(f"Failed to read input CSV: {e}")
        sys.exit(1)

    required_columns = {'system_prompt', 'question'}
    if not required_columns.issubset(df.columns):
        logging.error(f"Input CSV must contain columns: {required_columns}")
        sys.exit(1)

    total_requests = 0
    total_time = 0
    total_tokens = 0

    for index, row in df.iterrows():
        system_prompt = row['system_prompt']
        question = row['question']
        logging.info(f"Processing row {index + 1}: {question}")

        # Choose between chat and completion payload based on config
        if config['model'].get('type', 'chat') == 'chat':
            payload = {
                "model": config['model']['name'],
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                "temperature": config['model']['temperature'],
                "max_tokens": config['model']['max_tokens'],
                "stream": True
            }
        else:
            payload = {
                "prompt": f"{system_prompt}\n{question}",
                "model": config['model']['name'],
                "temperature": config['model']['temperature'],
                "max_tokens": config['model']['max_tokens'],
                "stream": True
            }

        headers = {}
        if 'headers' in config['api']:
            headers.update(config['api']['headers'])

        # Start timing
        start_time = time.time()
        time_to_first_token = None
        response_text = ""

        # Process the streamed response to capture time-to-first-token
        for line in send_streaming_request(config['api']['endpoint'], headers, payload):
            if line:  # Only process non-empty lines
                # Record time to first token
                if time_to_first_token is None:
                    time_to_first_token = time.time() - start_time

                # Accumulate response text for final token counting
                line_content = line.decode('utf-8').strip()
                if line_content:  # Ensure we only add meaningful content
                    response_text += line_content

        end_time = time.time()
        processing_time = end_time - start_time

        # Calculate tokens and performance metrics
        tokens_used = count_tokens(response_text)
        tokens_per_second = tokens_used / processing_time if processing_time > 0 else 0

        # Append to output CSV, now with model name included
        timestamp = datetime.utcnow().isoformat()
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                config['model']['name'],  # Model name added here
                system_prompt,
                question,
                response_text,
                f"{processing_time:.4f}",
                f"{time_to_first_token:.4f}" if time_to_first_token is not None else "N/A",
                tokens_used,
                f"{tokens_per_second:.2f}"
            ])

        # Construct the log message
        time_to_first_token_str = f"{time_to_first_token:.4f}s" if time_to_first_token is not None else "N/A"
        log_message = (
            f"Row {index + 1} processed in {processing_time:.2f}s, "
            f"Time to First Token: {time_to_first_token_str}, Tokens/Sec: {tokens_per_second:.2f}"
        )
        logging.info(log_message)

        # Update statistics
        total_requests += 1
        total_time += processing_time
        total_tokens += tokens_used

    # Log summary statistics
    if total_requests > 0:
        avg_time = total_time / total_requests
        avg_tps = total_tokens / total_time if total_time > 0 else 0
        logging.info("Processing Complete.")
        logging.info(f"Total Requests: {total_requests}")
        logging.info(f"Average Processing Time: {avg_time:.4f} seconds")
        logging.info(f"Average Tokens per Second: {avg_tps:.2f}")
        logging.info(f"Total Tokens Used: {total_tokens}")
    else:
        logging.warning("No successful requests processed.")

# ----------------------------
# Argument Parsing
# ----------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Model Comparison App")
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration YAML file.')
    parser.add_argument('--input_csv', type=str, help='Path to input CSV file.')
    parser.add_argument('--output_csv', type=str, help='Path to output CSV file.')
    parser.add_argument('--log_file', type=str, help='Path to log file.')
    parser.add_argument('--log_level', type=str, help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).')
    parser.add_argument('--model_name', type=str, help='Model name to use.')
    parser.add_argument('--temperature', type=float, help='Temperature setting for the model.')
    parser.add_argument('--max_tokens', type=int, help='Maximum tokens for the model response.')
    args = parser.parse_args()
    return args

# ----------------------------
# Main Entry Point
# ----------------------------

def main():
    args = parse_arguments()

    # Load configuration
    config = load_config(args.config)

    # Override configuration with command-line arguments if provided
    if args.input_csv:
        config['files']['input_csv'] = args.input_csv
    if args.output_csv:
        config['files']['output_csv'] = args.output_csv
    if args.log_file:
        config['files']['log_file'] = args.log_file
    if args.log_level:
        config['logging']['level'] = args.log_level
    if args.model_name:
        config['model']['name'] = args.model_name
    if args.temperature is not None:
        config['model']['temperature'] = args.temperature
    if args.max_tokens is not None:
        config['model']['max_tokens'] = args.max_tokens

    # Setup logging
    setup_logging(
        log_level=config['logging'].get('level', 'INFO'),
        log_format=config['logging'].get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        log_file=config['files'].get('log_file')
    )

    logging.info("Model Comparison App Started.")
    process_inputs(config)

if __name__ == "__main__":
    main()

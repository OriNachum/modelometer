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
import matplotlib.pyplot as plt
import psutil  # New library for CPU usage monitoring

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
# Performance Analysis Function
# ----------------------------

def analyze_model_performance(data, model_name, first_time_token):
    df = pd.DataFrame(data)
    
    # Generate summary statistics
    summary = df.describe()
    logging.info(f"Summary Statistics:\n{summary}")

    # Create plots for analysis
    plt.figure(figsize=(20, 8))

    # Boxplot for Tokens per Second
    plt.subplot(1, 4, 1)
    plt.boxplot(df['tokens_per_second'], vert=False)
    plt.title('Tokens Per Second Distribution')
    plt.xlabel('Tokens Per Second')

    # Boxplot for Time to First Token
    plt.subplot(1, 4, 2)
    plt.boxplot(df['time_to_first_token'], vert=False)
    plt.title('Time to First Token Distribution')
    plt.xlabel('Time to First Token (sec)')

    # Boxplot for CPU Usage
    plt.subplot(1, 4, 3)
    plt.boxplot(df['cpu_usage'], vert=False)
    plt.title('CPU Usage Distribution')
    plt.xlabel('CPU Usage (%)')

    # Add summary statistics as text on the fourth subplot
    plt.subplot(1, 4, 4)
    plt.axis('off')
    text = f"Summary Statistics\n\n{summary}\n\nFirst Time Run - Time to First Token: {first_time_token:.4f} sec"
    plt.text(0.5, 0.5, text, ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f"{model_name}.output.png")
    plt.close()

# ----------------------------
# Main Processing Function
# ----------------------------

def process_inputs(config):
    input_path = Path(config['files']['input_csv'])
    output_path = Path(config['files']['output_csv'])
    model_name = config['model']['name']

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
                'tokens_per_second',
                'cpu_usage'  # New field for CPU usage
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
    data_records = []  # Collect data for analysis

    first_time_token = None  # Variable to store the first time to first token

    for index, row in df.iterrows():
        system_prompt = row['system_prompt']
        question = row['question']
        logging.info(f"Processing row {index + 1}: {question}")

        # Choose between chat and completion payload based on config
        if config['model'].get('type', 'chat') == 'chat':
            payload = {
                "model": model_name,
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
                "model": model_name,
                "temperature": config['model']['temperature'],
                "max_tokens": config['model']['max_tokens'],
                "stream": True
            }

        headers = {}
        if 'headers' in config['api']:
            headers.update(config['api']['headers'])

        # Start timing and capture initial CPU usage
        start_time = time.time()
        start_cpu = psutil.cpu_percent(interval=None)  # Capture initial CPU usage
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

        # End timing and capture final CPU usage
        end_time = time.time()
        processing_time = end_time - start_time
        end_cpu = psutil.cpu_percent(interval=None)  # Capture final CPU usage
        avg_cpu_usage = (start_cpu + end_cpu) / 2  # Calculate average CPU usage for this run

        # Set the first run's time to first token and skip recording it in data for analysis
        if first_time_token is None:
            first_time_token = time_to_first_token
            continue

        # Calculate tokens and performance metrics
        tokens_used = count_tokens(response_text)
        tokens_per_second = tokens_used / processing_time if processing_time > 0 else 0

        # Append to output CSV
        timestamp = datetime.utcnow().isoformat()
        with open(output_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                timestamp,
                model_name,
                system_prompt,
                question,
                response_text,
                f"{processing_time:.4f}",
                f"{time_to_first_token:.4f}" if time_to_first_token is not None else "N/A",
                tokens_used,
                f"{tokens_per_second:.2f}",
                f"{avg_cpu_usage:.2f}"  # Record average CPU usage
            ])

        # Update data records for analysis
        data_records.append({
            'processing_time_sec': processing_time,
            'time_to_first_token': time_to_first_token,
            'tokens_used': tokens_used,
            'tokens_per_second': tokens_per_second,
            'cpu_usage': avg_cpu_usage  # Add CPU usage to data records
        })

        # Update statistics
        total_requests += 1
        total_time += processing_time
        total_tokens += tokens_used

    # Analyze and plot model performance, including first time to token
    analyze_model_performance(data_records, model_name, first_time_token)

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

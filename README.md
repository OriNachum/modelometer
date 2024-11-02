# Model'O'Meter

A simple command-line application to compare language models using the OpenAI API via Ollama. It processes inputs from a CSV file containing system prompts and questions, records processing time and average tokens per second for each response, and appends results to an output CSV file with comprehensive logging.

## Feature coverage
- [ ] Device specs
- [ ] Commit model stats 

## Models coverage
- [ ] Granite2
- [ ] Gemma 2 2B
- [ ] Llama3.2 1B
- [ ] Llama3.2 3B
- [ ] SmolLM 2 135M
- [ ] SmolLM 2 360M
- [ ] SmolLM 2 1.7B
- [ ] MobileLM 125M

## Example Output
![Granite Model Output](granite3-moe-3b.output.png)

## Features

- **CSV Input Handling**: Reads `system_prompt` and `question` from an input CSV file.
- **OpenAI API Integration via Ollama**: Sends requests to the Ollama API.
- **Performance Metrics Collection**: Measures processing time and calculates tokens per second.
- **Output Logging**: Appends results to an output CSV file.
- **Comprehensive Logging**: Logs operations with different levels (INFO, DEBUG, ERROR).
- **Error Handling**: Gracefully handles errors and continues processing.
- **Configurability**: Configure via `config.yaml` or command-line arguments.

## Requirements

- Python 3.7 or higher
- Install dependencies:

```bash
source .venv/bin/activate
```

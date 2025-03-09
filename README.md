# TweetyNet Song Detector

A Python tool for detecting and analyzing bird songs in WAV files using the TweetyNet model.

## Features

- Process multiple WAV files in batch
- Real-time progress tracking for each file
- Generate spectrograms and JSON results
- Skip already processed files
- Memory-efficient processing
- CUDA support for faster processing

## Requirements

- Python 3.x
- PyTorch
- librosa
- scipy
- tqdm
- torchvision

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/tweety-net-song-detector.git
cd tweety-net-song-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The script processes WAV files based on a list of paths provided in a text file. For each WAV file, it generates:
- A JSON file with detection results
- A spectrogram visualization

### Basic Usage

```bash
python process_wav_files.py --json_list path/to/file_list.txt
```

### Input File Format

The input text file should contain paths to JSON files with the `_f0.json` suffix. The script will look for corresponding WAV files by removing this suffix and adding `.wav`. For example:

```
data/folder1/recording1_f0.json
data/folder2/recording2_f0.json
```

Will process:
```
data/folder1/recording1.wav
data/folder2/recording2.wav
```

### Optional Arguments

- `--model_path`: Path to custom model directory (optional)

### Output

For each WAV file, the script generates:
- `{wav_name}_results.json`: Detection results
- `specs/{wav_name}_detection.png`: Spectrogram visualization

## Progress Tracking

The script provides detailed progress information:
- Overall progress through all files
- Per-file processing status
- Real-time processing speed
- Estimated time remaining
- Processing statistics

## License

[Add your chosen license here] 
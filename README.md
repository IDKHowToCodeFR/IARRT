# IARRT: Idiom-Aware Retrieval and Robust Translation

NLP project for idiom detection, retrieval, and translation using Transformers.

## Features
- **Idiom Detection**: Identify idioms in text using token classification.
- **Idiom Retrieval**: Retrieve idiom definitions/meanings from a dictionary.
- **Robust Translation**: Translate sentences containing idioms with high accuracy.
- **Baseline Comparison**: Tools to compare performance against standard baselines.

## Structure
- `main.py`: Entry point for end-to-end pipeline.
- `train_detector.py`: Training script for idiom detection.
- `train_translator.py`: Training script for translation model.
- `inference.py`: Run inference on new text.
- `data/`: Data loading and idiom dictionary.
- `retrieval/`: Idiom retrieval logic.
- `utils/`: Evaluation and detection utilities.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare data in `data/` folder.
3. Run training:
   ```bash
   python train_detector.py
   python train_translator.py
   ```

## Usage
Run full pipeline:
```bash
python main.py --input "Your sentence with idiom"
```

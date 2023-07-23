# Semantic Features of Object Concepts Generated With GPT-3
This repository contains all needed code to generate, preprocess, norm and evaluate semantic feature norms from GPT-3.

## Installation
```
pip install -r requirements.txt
pip install -e .
```

## Project structure
```
.
├── analysis                                    # All analyses from the paper
│   ├── category_structure                      # Analysis of category structure
│   ├── distribution                            # Analysis of featuer distribution
│   |── similarity_prediction                   # Analysis of human similarity judgement prediction
│   |── stats                                   # Statistics about feature norms
│   |── dimensions                              # Most frequent features per dimension
│   |── quality_labling                         # Human judgements about sensible features

|── data
|   |── gpt_3_feature_norm                      # Generated features
|       |── decoded_answers.csv                 # Preprocessed and normed features -> final feature norm
|       |── encoded_answers_openai.csv          # Raw answers from GPT-3
|       |── feature_object_matrix.csv           # Feature-Concept frequency matrix based on the final feature norm
│   |── train                                   # Priming examples for 30 runs
│   all_concepts.csv                            # Questions to retrieve new features 

|── scripts                                     # Different scripts
|   |── data_taking                             # All code needed to generate features with GPT-3
|   |── decoding                                # All code needed to preprocess and norm the feature norm
|   |── vectorization                           # Create feature-concept matrices
```

### Feature generation
To generate features with GPT-3, you need an API token from OpenAI.
```
cd bin
OPENAI_API_KEY python create_feature_norm.py --output_dir=. --train_dir=train --retrival_path=priming.csv
```

### Decoding
```
cd bin
python decode.py --answers=feature_norm_from_gpt.csv --output=. --parallel
```

### Evaluation
To reproduce the analyses and figures, you can run the notebooks and scripts in the analysis directory.

## Citation


## Prices
Prompting - 350 Tokens
Completion - 70 Tokens 

GPT-4
0.03 pro Prompt 
0.06 fuer completion 

10 Runs - 317
317 x 10 x 350 tokens / 1000 * 0.03 = 33
317 x 10 x 70 token / 1000 * 0.06 = 13

30 Runs - 317
317 x 10 x 350 tokens / 1000 * 0.03 = 200
317 x 10 x 70 token / 1000 * 0.06 = 39

GPT-3
0.02 pro 1000 Prompt 

30 Runs
317 x 420 x 30 / 1000 * 0.02 = 80



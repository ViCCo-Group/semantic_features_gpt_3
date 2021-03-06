# Semantic Features of Object Concepts Generated With GPT-3
This repository contains all needed code to generate, preprocess, norm and evaluate semantic feature norms from GPT-3.

## Installation
```
pip install -r requirements.txt
```

## Project structure
```
.
├── analysis                                    # All analyses from the paper
│   ├── category_structure                      # Analysis of category structure
│   ├── distribution                            # Analysis of featuer distribution
│   |── similarity_prediction                   # Analysis of human similarity judgement prediciton
│   |── variance                                # Analysis of variance partitioning
│   |── stats                                   # Statistics about feature norms
|── data
|── |── gpt_3_feature_norm                          # Generated features
|       |── decoded_answers.csv                     # Preprocessed and normed features -> final feature norm
|   |── encoded_answers_openai.csv              # Raw answers from GPT-3
|── data_taking                                 # All code needed to generate features with GPT-3
│   |── train                                   # Priming examples for 30 runs
│   |── all_concepts.csv                        # Questions to retrieve new features 
|── decoding                                    # All code needed to preprocess and norm the feature norm

```

### Data generation
To generate features with GPT-3, you need an API token from OpenAI.
```
OPENAI_API_KEY python run_openai.py
```

### Decoding
```
```

### Evaluation
To reproduce the analyses and figures, you can run the notebooks in the analysis directory.

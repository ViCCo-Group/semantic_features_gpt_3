from copy import deepcopy
from utils.data import load_cslb, load_gpt, load_mcrae
from copy import deepcopy

group_to_one_concept = True
duplicates = True

mc = load_mcrae(group_to_one_concept, duplicates)
cslb = load_cslb(group_to_one_concept)
gpt = load_gpt(1, group_to_one_concept, 1, duplicates, 'mcrae_priming', 'gpt3-davinci', 10, 1854)

# All THINGS concepts
all_things_concepts = set(gpt['concept_id'].unique())
with open('your_file.txt', 'w') as f:
    for line in all_things_concepts:
        f.write(f"{line}\n")

# Test set - Intersection of all feature norms 
test_concepts = deepcopy(all_things_concepts)
for feature_norm in [mc, cslb]:
    test_concepts = test_concepts.intersection(set(feature_norm['concept_id']))
with open('your_file.txt', 'w') as f:
    for line in test_concepts:
        f.write(f"{line}\n")

# Validation set - THINGS concepts without intersection 
val_concepts = all_things_concepts.difference(test_concepts)
with open('your_file.txt', 'w') as f:
    for line in val_concepts:
        f.write(f"{line}\n")



   
# OnT (Language Models as Ontology Encoder)

## Project Overview
OnT is a language model-based framework for ontology embeddings, enabling effective representation of concepts as points in hyperbolic space and axioms as hierarchical relationships between concepts. Built upon the [HierarchyTransformer](https://github.com/KRR-Oxford/HierarchyTransformers), this implementation provides enhanced capabilities for ontological reasoning through specialized embedding techniques such as concept rotation, transition, and existential quantifier representation. The model has been trained on various biomedical ontologies including GO, GALEN, and ANATOMY datasets. For the test of geometric-embedding methods, please refer to the [BoxSquaredEL](https://github.com/KRR-Oxford/BoxSquaredEL) and [TransBox](https://github.com/HuiYang1997/TransBox).

## Features
- Hyperbolic embeddings for ontology concept encoding
- Modeling of hierarchical relationships between concepts
- Support for role embeddings as rotations over hyperbolic spaces

## Project Structure
**The data and models folders should be downloaded and unzipped to the root directory. The Google Drive links are all anonymous.**
- `OnT.py`: Main model implementation containing the OntologyTransformer class
- `data/`: Contains training and testing data, download from [here](https://drive.google.com/file/d/1pqHKdj0R-M45ny44xwhsL3n3zeZORsvz/view?usp=drive_link)
- `models/`: Stores pre-trained and fine-tuned models, download from [here](https://drive.google.com/file/d/1t9xWcLHoEE55F0bOPMCw5jltWBxHc2vR/view?usp=drive_link)
- `normalization/`: Scripts for normalizing the EL part of a given ontology. See `normalization/Readme.md` for details.
- `HierarchyTransformers/`: Repository of [HiT](https://github.com/KRR-Oxford/HierarchyTransformers), to be installed through GitHub

## Installation
Please first install [HiT](https://github.com/KRR-Oxford/HierarchyTransformers) through GitHub in this path, following the instructions of their repository. Our implementation is currently also fixed on `sentence-transformers=3.4.0.dev0`.


## Usage
Then load the model and use it for inference or training as follows.

```python
import torch
from OnT import OntologyTransformer

# Initialize the ont model
# OnTr is the full version, OnT is the simple version trained without embedding
path = "models/inferences/OnTr-all-MiniLM-L12-v2-GALEN" 
ont = OntologyTransformer.load(path)

# entity names to be encoded.
entity_names = ["continuant", "occurrent", "independent continuant", "process"]

# get the entity embeddings
entity_embeddings = ont.encode_concept(entity_names)

# role sentences to be encoded.
role_sentences = ["application attribute", "attribute", "chemical modifier", "chemical process modifier attribute"]

# get the role embeddings, consist of the rotation and scaling
role_rotations, role_scalings = ont.encode_roles(role_sentences)

```


For training, run the following command. Remember to update the **dataset_path** and **dataset_name** in `config.yaml` :
```
python train_ont.py -c config.yaml
```

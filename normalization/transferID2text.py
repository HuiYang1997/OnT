import numpy as np
import json
import os
from random import sample
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_names(concept_file, role_file):
    with open(concept_file, 'r') as f:
        concept_names = json.load(f)
    with open(role_file, 'r') as f:
        role_names = json.load(f)
    return concept_names, role_names

def process_npy_file(file_path, concept_names, role_names, kind):
    # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Process the data to generate sentence pairs
    # (This is where you will apply your specific formula)
    sentence_pairs = []
    conjunction_paris = []
    existential_pairs = []


    # Example logic (to be replaced with actual processing):
    for item in data:
        # Generate sentence pairs based on item and names
        if kind == 'nf1':
            # in this case, data is a matrix of the shape (n, 2), where n is the number of sentences, and 2 is the number of concepts in each sentence
            sentence1 = concept_names[str(item[0])]
            sentence2 = concept_names[str(item[1])]
        elif kind == 'nf2':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqcap 1 \sqsubseteq 2
            sentence1 = f"{concept_names[str(item[0])]} and {concept_names[str(item[1])]}"
            sentence2 = concept_names[str(item[2])]
            conjunction_paris.append((concept_names[str(item[0])], concept_names[str(item[1])]))
        elif kind == 'nf3':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqsubseteq \exists 1. 2
            sentence1 = concept_names[str(item[0])]
            sentence2 = f"{role_names[str(item[1])]} some {concept_names[str(item[2])]}"
            existential_pairs.append((role_names[str(item[1])], concept_names[str(item[2])]))
        elif kind == 'nf4':
            # in this case, data is a matrix of the shape (n, 3), the axioms means \exists 0. 1 \sqsubseteq  2
            role_name = role_names[str(item[0])]
            sentence1 = f"{role_name} some {concept_names[str(item[1])]}"
            sentence2 = concept_names[str(item[2])]
            existential_pairs.append((role_name, concept_names[str(item[1])]))
        else:
            raise ValueError(f"Unknown axiom kind: {kind}")
        sentence_pairs.append((sentence1, sentence2))
    
    return sentence_pairs, conjunction_paris, existential_pairs

def process_test_val_data(file_path, concept_names, role_names, kind):
     # Load the .npy file
    data = np.load(file_path, allow_pickle=True)
    
    # Process the data to generate sentence pairs
    # (This is where you will apply your specific formula)
    sentences = []
    answer_ids = []
    roles, cons = [], []
    con1s, con2s = [], []

    # Example logic (to be replaced with actual processing):
    for item in data:
        # Generate sentence pairs based on item and names
        if kind == 'nf1':
            # in this case, data is a matrix of the shape (n, 2), where n is the number of sentences, and 2 is the number of concepts in each sentence
            sentences.append(concept_names[str(item[0])])
            answer_ids.append(int(item[1]))
        elif kind == 'nf2':
            sentences.append(f"{concept_names[str(item[0])]} and {concept_names[str(item[1])]}")
            answer_ids.append(int(item[2]))
            con1s.append(concept_names[str(item[0])])
            con2s.append(concept_names[str(item[1])])
        elif kind == 'nf3':
            # in this case, data is a matrix of the shape (n, 3), the axioms means 0 \sqsubseteq \exists 1. 2
            sentences.append(f"{role_names[str(item[1])]} some {concept_names[str(item[2])]}")
            answer_ids.append(int(item[0]))
            roles.append(role_names[str(item[1])])
            cons.append(concept_names[str(item[2])])
        elif kind == 'nf4':
            role_name = role_names[str(item[0])]
            # in this case, data is a matrix of the shape (n, 3), the axioms means \exists 0. 1 \sqsubseteq  2
            roles.append(role_name)
            sentences.append(f"{role_name} some {concept_names[str(item[1])]}")
            answer_ids.append(int(item[2]))
            cons.append(concept_names[str(item[1])])
        else:
            raise ValueError(f'Unknown kind: {kind}')
    
    return sentences, answer_ids, roles, cons, con1s, con2s

def load_val_test(data_dir, out_dir, load_type = 'val'):
    concept_file = f'{data_dir}/concept_names.json'
    role_file = f'{data_dir}/role_names.json'
    inverse_role_file = f'{data_dir}/role_inverse_mapping.json'    
    
    # Load names
    concept_names, role_names = load_names(concept_file, role_file) 

    # process test or val data
    processed_data = {'query_sentences':{}, 'answer_ids':{}}
    data_dir = f'{data_dir}/ont_tmp/{load_type}'
    file_names_to_process = [f'nf{i}.npy' for i in range(1, 5)]
    for file_name in file_names_to_process:
        file_path = os.path.join(data_dir, file_name)
        sentences, answer_ids, roles, cons, con1s, con2s = process_test_val_data(file_path, concept_names, role_names, kind = file_name[:-4])
        processed_data['answer_ids'][file_name[:-4]] = answer_ids
        if file_name == 'nf1.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":s} for s in sentences]
        elif file_name == 'nf2.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":sentences[i], "con1":con1s[i], "con2":con2s[i]} for i in range(len(sentences))]
        elif file_name == 'nf3.npy' or file_name == 'nf4.npy':
            processed_data['query_sentences'][file_name[:-4]] = [{"name":sentences[i], "role":roles[i], "con":cons[i]} for i in range(len(sentences))]
        else:
            print('error')

    with open(os.path.join(out_dir, f'{load_type}.json'), 'w') as f:
        json.dump(processed_data, f, indent=8)

    return


def load_train(data_dir, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    concept_file = f'{data_dir}/concept_names.json'
    role_file = f'{data_dir}/role_names.json'
    
    # Load names
    concept_names, role_names = load_names(concept_file, role_file)
    
    # Translate data
    file_names_to_process = [f'nf{i}.npy' for i in range(1, 5)]

    conjunction_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    existential_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
    sentence_pair_dict = {"nf1": [], "nf2": [], "nf3": [], "nf4": []}
  
    nf_data_dir = f'{data_dir}/ont_tmp/train'
    for file_name in file_names_to_process:
        file_path = os.path.join(nf_data_dir, file_name)
        sent_pairs, conj_pairs, exist_pairs = process_npy_file(file_path, concept_names, role_names, kind = file_name[:-4])
        sentence_pair_dict[file_name[:-4]]+=sent_pairs
        conjunction_pair_dict[file_name[:-4]]+=conj_pairs
        existential_pair_dict[file_name[:-4]]+=exist_pairs
    
    
    # save each nf data separately
    all_data = []
    for key, sentence_pairs in sentence_pair_dict.items():
        roles = existential_pair_dict[key]
        conj_pairs = conjunction_pair_dict[key]
        output_file = f'{out_dir}/train_{key}.jsonl'
        with open(output_file, 'w') as out_file:
            for ind, pair in enumerate(sentence_pairs):
                # randomly select 10 names from concept_names as negative samples
                if key == 'nf3':
                    role_name = f"{roles[ind][0]} some "
                else:
                    role_name = ""
                negative_samples = sample(list(concept_names.values()), k=10)
                negative_samples = [role_name + negs for negs in negative_samples]
                data_for_hit = {"child": pair[0], "parent": pair[1], "negative": negative_samples}
                all_data.append(data_for_hit)

                if key == 'nf1':
                    data_ours = data_for_hit
                elif key == 'nf2':
                    data_ours = { "con1":conj_pairs[ind][0], "con2":conj_pairs[ind][1], "parent":pair[1], "negative": negative_samples}
                elif key == 'nf3':
                    data_ours = {"atomic":pair[0], "role":roles[ind][0], "con":roles[ind][1], "negative": negative_samples}
                elif key == 'nf4':
                    data_ours = {"atomic":pair[1], "role":roles[ind][0], "con":roles[ind][1], "negative": negative_samples}
                else:
                    raise ValueError(f"Unknown axiom kind: {key}")

                out_file.write(json.dumps(data_ours) + "\n")
    
    # save all data
    output_file = f'{out_dir}/train.jsonl'
    with open(output_file, 'w') as out_file:
        for data in all_data:
            out_file.write(json.dumps(data) + "\n")
    
    # save conjunction pairs
    output_file = f'{out_dir}/train_conj.jsonl'
    with open(output_file, 'w') as out_file:
        # Write sentence pairs to the output file
        for key, conj_pairs in conjunction_pair_dict.items():
            for pair in conj_pairs:
                data = {"Concept": f"{pair[0]} and {pair[1]}", "con1": pair[0], "con2": pair[1]}
                out_file.write(json.dumps(data) + "\n")
    

    # save existential pairs
    output_file = f'{out_dir}/train_exist.jsonl'
    with open(output_file, 'w') as out_file:
        # Write sentence pairs to the output file
        for key, existential_pairs in existential_pair_dict.items():
            for pair in existential_pairs:
                data = {"Concept": f"{pair[0]} some {pair[1]}", "role": pair[0], "con": pair[1]}
                out_file.write(json.dumps(data) + "\n")
    
    # save all entities
    output_file = f'{out_dir}/concept_names.json'
    with open(output_file, 'w') as out_file:
        json.dump(concept_names, out_file, indent=4)

    # save all roles
    output_file = f'{out_dir}/role_names.json'
    with open(output_file, 'w') as out_file:
        json.dump(role_names, out_file, indent=4)

    
    return

if __name__ == "__main__":
    all_data_dir = '/data/Hui/HiT_new/HierarchyTransformers/data_new'
    load_train(all_data_dir, task='OnT')
    load_val_test(all_data_dir, task='OnT', load_type = 'val')
    load_val_test(all_data_dir, task='OnT', load_type = 'test')
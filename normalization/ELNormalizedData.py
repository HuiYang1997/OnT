import json
import os
import re
import shutil
import sys

import numpy as np
from typing import Dict, List, Optional
from yacs.config import CfgNode

from deeponto.onto import Ontology, OntologyVerbaliser, OntologyNormaliser, OntologyReasoner
from transferID2text import load_train, load_val_test

# fix random seed
np.random.seed(42)

# Function to convert camel case to a spaced phrase
def camel_case_to_spaced(phrase):
    phrase = phrase.split('#')[-1]

    # Split the phrase into maximal segments that containing only uppercase or lowercase letters(or any other characters other than letters)
    segments = re.findall(r'[A-Z]+|[^A-Z]+', phrase)
    
    # Process each segment
    second_phrase = ''
    for segment in segments:
        # if the last element of segment is a uppercase letter, transform it to lowercase, and add a space before the last element
        # exclude romane numbers
        numbers_romain = {'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX'}
        if segment[-1].isupper():
            if len(segment)==1:
                segment = segment.lower()
            elif len(segments)>1 and segment not in numbers_romain:
                segment = segment[:-1] + ' ' + segment[-1].lower()
        else:
            segment = segment + ' '
        second_phrase += segment
    
    second_phrase = second_phrase.strip()

    print(f"{phrase} -> {second_phrase}")
    
    return second_phrase

class ELNormalizedData:
    """
    Create EL normalized dataset for embeddings, with the name of new introduced atomic concepts provided.
    There are two subfolders: OnT and Prediction.
    For OnT, only orginal axioms is saved.
    For Prediction, both original axioms and inferred axioms are saved.
    """
    
    def __init__(self):
        self.concept_names: Dict[str, tuple[int, str]] = {}  # name to (ind, iri) mapping for concepts
        self.role_names: Dict[str, tuple[int, str]] = {}     # name to (ind, iri) mapping for roles
        self.nf1: List[list[int]] = []  # Normal Form 1 axioms A\sqsubseteq B
        self.nf2: List[list[int]] = []  # Normal Form 2 axioms A\sqcap B\sqsubseteq C
        self.nf3: List[list[int]] = []  # Normal Form 3 axioms A\sqsubseteq \exists r. B
        self.nf4: List[list[int]] = []  # Normal Form 4 axioms \exists r. A\sqsubseteq B


        self.nf1_org: List[list[int]] = []  # Normal Form 1 axioms A\sqsubseteq B
        self.nf2_org: List[list[int]] = []  # Normal Form 1 axioms A\sqsubseteq B
        self.nf3_org: List[list[int]] = []  # Normal Form 1 axioms A\sqsubseteq B
        self.nf4_org: List[list[int]] = []  # Normal Form 1 axioms A\sqsubseteq B

        self.role_inclusion: List[list[int]] = []  # role_inclusion axioms r\sqsubseteq t
        self.role_chain: List[list[int]] = []  # role_chain axioms r\circ s \sqsubseteq t

        self.inference_axioms: List[list[int]] = []  # inferred axioms of the form A⊑B
        self.ind_normalized = 1
        self.output_dir = None
    
    def initial_atomic_names(self, ont, vocab) -> None:
        i = 0
        for iri in ont.owl_classes:
            if vocab[iri] in self.concept_names:
                print(f"{iri} repeat to {self.concept_names[vocab[iri]]}, name: {vocab[iri]}")
            else:
                self.concept_names[vocab[iri]] = (i, iri)
                i += 1
        
        j = 0
        for iri in ont.owl_object_properties:
            if vocab[iri] in self.role_names:
                print(f"{iri} repeat to {self.role_names[vocab[iri]]}, name: {vocab[iri]}")
            else:
                self.role_names[vocab[iri]] = (j, iri)
                j += 1
    
    def save(self) -> None:
        """
        Save the ontology data to files in the specified directory.
        
        Args:
            output_dir (str): Directory to save the data files
        """
        # Create output directory if it doesn't exist
        for folder in ["prediction", "ont_tmp"]:
            for subfolder in ["train", "test", "val"]:
                os.makedirs(os.path.join(self.output_dir, folder, subfolder), exist_ok=True)     
  
        # Save concept names
        concepts_name = {str(v[0]): k for k, v in self.concept_names.items()}
        with open(os.path.join(self.output_dir, "concept_names.json"), "w") as f:
            json.dump(concepts_name, f, indent=3)
        
        # Save role names
        roles_name = {str(v[0]): k for k, v in self.role_names.items()}
        with open(os.path.join(self.output_dir,  "role_names.json"), "w") as f:
            json.dump(roles_name, f, indent=3)
        

        classes = {str(v[0]): v[1] for v in self.concept_names.values()}
        relations = {str(v[0]): v[1] for v in self.role_names.values()}
        for folder in ["prediction", "ont_tmp"]:
            with open(os.path.join(self.output_dir, folder, "classes.json"), "w") as f:
                json.dump(classes, f, indent=3)

            with open(os.path.join(self.output_dir, folder, "relations.json"), "w") as f:
                json.dump(relations, f, indent=3)
                
        # Handle self.nf1_org separately
        print(f"nf1_org: {len(self.nf1_org)}, nf2_org: {len(self.nf2_org)}, nf3_org: {len(self.nf3_org)}, nf4_org: {len(self.nf4_org)}")
        print(f"nf1: {len(self.nf1)}, nf2: {len(self.nf2)}, nf3: {len(self.nf3)}, nf4: {len(self.nf4)}")

        # First partition nfi_org
        nf_org_splits = []
        for i, nf_org in enumerate([self.nf1_org, self.nf2_org, self.nf3_org, self.nf4_org], 1):
            nf_org_array = np.array(nf_org)
            np.random.shuffle(nf_org_array)
            total_len = len(nf_org_array)
            train_len = int(0.8 * total_len)
            val_len = int(0.1 * total_len)
            test_len = total_len - train_len - val_len
            
            train_org = nf_org_array[:train_len]
            val_org = nf_org_array[train_len:train_len+val_len]
            test_org = nf_org_array[train_len+val_len:]
            nf_org_splits.append((train_org, val_org, test_org))
            print(f"nf{i}_org splits: train={len(train_org)}, val={len(val_org)}, test={len(test_org)}")

            # saving only orginal axims for OnT for training and val
            np.save(os.path.join(self.output_dir, "ont_tmp", "train", f"nf{i}.npy"), train_org)
            np.save(os.path.join(self.output_dir, "ont_tmp", "val", f"nf{i}.npy"), val_org)


        # Then partition nfi and combine with nfi_org
        for i, nf in enumerate([self.nf1, self.nf2, self.nf3, self.nf4], 1):
            nf_array = np.array(nf)
            np.random.shuffle(nf_array)
            total_len = len(nf_array)
            train_len = int(0.8 * total_len)
            val_len = int(0.1 * total_len)
            test_len = total_len - train_len - val_len
            
            train_set = nf_array[:train_len]
            val_set = nf_array[train_len:train_len+val_len]
            test_set = nf_array[train_len+val_len:]
            
            # Combine with corresponding nfi_org splits
            train_org, val_org, test_org = nf_org_splits[i-1]
            train_combined = train_set if not train_org.size else np.concatenate([train_set, train_org])
            val_combined = val_set if not val_org.size else np.concatenate([val_set, val_org])
            test_combined = test_set if not test_org.size else np.concatenate([test_set, test_org])
            
            print(f"nf{i} final splits: train={len(train_combined)}, val={len(val_combined)}, test={len(test_combined)}")
            
            # Save combined sets
            np.save(os.path.join(self.output_dir, 'prediction', 'train', f"nf{i}.npy"), train_combined)
            np.save(os.path.join(self.output_dir, 'prediction', 'val', f"nf{i}.npy"), val_combined)
            np.save(os.path.join(self.output_dir, 'prediction', 'test', f"nf{i}.npy"), test_combined)

            # save the same test set to ont_tmp folder
            np.save(os.path.join(self.output_dir, 'ont_tmp', 'test', f"nf{i}.npy"), test_combined)
            
        
        # saving role inclusion as np.array in prediction folder
        np.save(os.path.join(self.output_dir, 'prediction', 'train', 'role_inclusion.npy'), self.role_inclusion)

        #saving role_chain as np.array in prediction folder
        np.save(os.path.join(self.output_dir, 'prediction', 'train', 'role_chain.npy'), self.role_chain)

        # saving all concept ids as np.array in prediction folder
        np.save(os.path.join(self.output_dir, 'prediction', 'train', 'class_ids.npy'), np.array(list(range(len(self.concept_names)))))

        # saveing a  single array [0] to top.npy
        np.save(os.path.join(self.output_dir, 'prediction', 'train', 'top.npy'), np.array([0]))

        # saving a empty array [] to disjoint.npy
        np.save(os.path.join(self.output_dir, 'prediction', 'train', 'disjoint.npy'), np.array([]))

        # transfer ont_tmp to textual data for OnT
        self.transferID2text()
        
        print("Finished.")

        return

    def transferID2text(self):
        """
        Transfer the ID to text in the prediction folder.
        """
        # transfer ont_tmp to textual data for OnT
        input_dir = os.path.join(self.output_dir)
        output_dir = os.path.join(self.output_dir, 'OnT')
        load_train(input_dir, output_dir)
        load_val_test(input_dir, output_dir, 'val')
        load_val_test(input_dir, output_dir, 'test')

        # delete ont_tmp folder
        shutil.rmtree(os.path.join(self.output_dir, 'ont_tmp'))
        
        print("Finished.")

        return

    
    def update_from_subsumptions(self, child:CfgNode, parent:CfgNode) -> None:
        """
        Update the ontology data by adding subsumptions.
        
        Args:
            subsumptions (List[str]): List of subsumptions to add
        """
        if parent['type']=='AND':
            for C_i in parent['classes']:
                self.update_from_subsumptions(child, C_i)
            return
        elif parent['type']=='EX.':
            D = parent['class']
            D_id = self.update_from_complex_concepts(D)
            C_id = self.update_from_complex_concepts(child)

            role_name = parent['property']['verbal']
            role_id = self.role_names[role_name][0]

            self.nf3_org.append([C_id, role_id, D_id])
        else:
            assert parent['type']=='IRI'
            parent_id = self.concept_names[parent['verbal']][0]
            if child['type']=='IRI':
                child_id = self.concept_names[child['verbal']][0]
                self.nf1_org.append([child_id, parent_id])
            elif child['type']=='AND':
                child_name = child['verbal']
                child_id_first = self.update_from_complex_concepts(child['classes'][0])
                if len(child['classes']) > 2:
                    child_id_rest = self.update_from_complex_concepts(child['classes'][1:])
                else:
                    assert len(child['classes']) == 2
                    child_id_rest = self.update_from_complex_concepts(child['classes'][1])
                self.nf2_org.append([child_id_first, child_id_rest, parent_id])
            else:
                assert child['type']=='EX.'
                sub_child_id = self.update_from_complex_concepts(child["class"])

                role_name = child["property"]['verbal']
                role_id = self.role_names[role_name][0]
                self.nf4_org.append([role_id, sub_child_id, parent_id])
    
    
    def update_from_equivalent(self, child:CfgNode, parent:CfgNode) -> None:
        """
        Update the ontology data by adding equivalent axioms.
        
        Args:
            equivalents (List[str]): List of equivalent axioms to add
        """
        self.update_from_subsumptions(child, parent)
        self.update_from_subsumptions(parent, child)

    
    def update_concept_names(self, concept_name:str) -> int:
        if concept_name not in self.concept_names:
            current_ind = len(self.concept_names)
            self.concept_names[concept_name] = (current_ind, f"N{self.ind_normalized}")
            self.ind_normalized += 1
        
        return self.concept_names[concept_name][0]

    
    def update_from_complex_concepts(self, complex_concept:CfgNode | list[CfgNode]) -> None:
        """
        Update the ontology data as the EL normalization axioms corresponding to given complex concepts.

        Args:
            complex_concept (CfgNode): Complex concept to add
            given_id (Optional[int]): give ID of the complex concept, only occur when updated from equivalent axioms
        """
        if isinstance(complex_concept, CfgNode) and complex_concept["type"] == 'IRI':
            return self.concept_names[complex_concept["verbal"]][0]

        if isinstance(complex_concept, CfgNode):
            complex_name = complex_concept["verbal"]
        else:
            all_names = [node["verbal"] for node in complex_concept]
            complex_name = ' and '.join(all_names)

        if complex_name not in self.concept_names:
            current_id = self.update_concept_names(complex_name)
            
            if isinstance(complex_concept, CfgNode) and complex_concept["type"] == 'EX.':
                role_name = complex_concept["property"]['verbal']
                role_id = self.role_names[role_name][0]

                subconcept_id = self.update_from_complex_concepts(complex_concept["class"])

                # add C\sqsubseteq \exists r. B
                self.nf3.append([current_id, role_id, subconcept_id])
                
                # add \exists r. B\sqsubseteq C 
                self.nf4.append([role_id, subconcept_id, current_id])

            else:
                if isinstance(complex_concept, list):
                    subconcepts = complex_concept
                elif complex_concept["type"] == 'AND':
                    subconcepts = complex_concept["classes"]
                else:
                    raise ValueError("Complex concept type not supported")
                
                if len(subconcepts) == 1:
                    return self.update_from_complex_concepts(subconcepts[0])
                
                subconcept_names = [subconcept["verbal"] for subconcept in subconcepts]    
                subconcept_ids = [self.update_from_complex_concepts(subconcept) for subconcept in subconcepts]

                for sub_id in subconcept_ids:
                    self.nf1.append([current_id, sub_id])
                 
                # add C_i \sqcap C_{i+1:} \sqsubseteq C_{i:} 
                previous_id = current_id
                for i, the_id in enumerate(subconcept_ids):
                    if i+1 < len(subconcept_ids):
                        verbal = " and ".join(subconcept_names[i+1:])
                        rest_id = self.update_concept_names(verbal)
                    else:
                        break
                    self.nf2.append([the_id, rest_id, previous_id])
                    previous_id = rest_id
        
        return self.concept_names[complex_name][0]
    
    def load(self, ontology_path, output_dir):
        """
        Load an ontology
        
        Args:
            ontology_path (str): Path to the input ontology file
            output_dir (str): Directory to save the output files
        """
        # Load the ontology
        ont = Ontology(ontology_path)
    
        # Initialize verbalizer for naming new concepts
        verbalizer = OntologyVerbaliser(ont,add_quantifier_word=True)
        if not verbalizer.vocab:
            print("\n" + "="*80)
            print("WARNING: The vocabulary of verbalizer is empty.")
            print("This will be automatically updated using camel case vocabulary from concept or property IRIs.")
            print("Please verify the resulting vocabulary for correctness.")
            print("="*80)
            
            proceed = input("\nEnter 'y' to proceed automatically, or any other key to manually confirm each update: ")
            
            # Update the vocab by the camel vocab of concept or property IRIs
            for iri in ont.owl_classes:
                verbalizer.update_entity_name(iri, camel_case_to_spaced(iri))
            for iri in ont.owl_object_properties:
                verbalizer.update_entity_name(iri, camel_case_to_spaced(iri))
                
            if proceed.lower() != 'y':
                print(f"\nVocabulary has been updated with {len(verbalizer.vocab)} entries.")
                confirm = input("Enter 'OK' to continue with these updates: ")
                if confirm.strip().upper() != "OK":
                    raise ValueError("Operation cancelled by user. Please provide a valid vocabulary and try again.")
        
        self.initial_atomic_names(ont, verbalizer.vocab)
        return ont, verbalizer
    
    def create_dataset(self, ont, verbalizer):
        # preprocess it to EL, normalize it and save the results.

        # Normalize the ontology
        normalizer = OntologyNormaliser()
        processed_ont =  normalizer.preprocess_ontology(ont)
        processed_axioms = []
        processed_axioms.extend(list(processed_ont.getAxioms()))

        for ont in processed_ont.getImportsClosure():
            processed_axioms.extend(list(ont.getAxioms()))

        # Process normalized axioms to extract concept names
        for axiom in processed_axioms:
            # determine the type of the axiom
            axiom_type = verbalizer.onto.get_axiom_type(axiom)
            if axiom_type == 'SubClassOf':
                # print(axiom)
                try:
                    verb_result = verbalizer.verbalise_class_subsumption_axiom(axiom)
                except:
                    print("Failed to verbalise axioms:" ,axiom)
                self.update_from_subsumptions(*verb_result)
            elif axiom_type == 'EquivalentClasses':
                try:
                    verb_result = verbalizer.verbalise_class_equivalence_axiom(axiom)
                except:
                    print("Failed to verbalise axioms:" ,axiom)

                self.update_from_equivalent(*verb_result)
            else:
                continue


    def get_role_inclusion(self, ont, verbalizer):
        # get the role inclusion axioms r\sqsubseteq t and save it to self.role_inclusion

        for ont_axiom in ont.get_subsumption_axioms("ObjectProperties"):
            try:
                sub_verbal, sup_verbal = verbalizer.verbalise_object_property_subsumption_axiom(ont_axiom)
            except:
                print("Failed to verbalise axioms:" ,ont_axiom)
                continue
            
            sub_id = self.role_names[sub_verbal['verbal']][0]
            sup_id = self.role_names[sup_verbal['verbal']][0]
            self.role_inclusion.append([sub_id, sup_id])

    def main(self, ont_path, output_dir):
        self.output_dir = output_dir

        ont, verbalizer = self.load(ont_path, output_dir)
        self.get_role_inclusion(ont, verbalizer)
        self.create_dataset(ont, verbalizer)
        self.save()

if __name__ == '__main__':
    input_ont_path = "your_ont_path"
    output_dir = "your_output_dir"

    # run the following code to generate the EL normalized dataset
    ELNormalizedData().main(input_ont_path, output_dir)

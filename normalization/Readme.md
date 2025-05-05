# Name preserving Normalization of EL ontologies



The normalization algorithm aims to extract the EL part of an ontology, and transfer it to a normalized form. The main difference between the standard normalziation progress is that we assign names to the new atomic concepts $N_C$ based on the verbalizations of the corresponding complex concepts $C$.

### Usage

Use the script `ELNormalizedData.py` to normalize an ontology by the following command:

```bash
python ELNormalizedData.py -i <input_ontology_path> -o <output_directory>
```

It will output two subfolders: `OnT` and `Prediction`, the first one only contains the textual training data, which is used for training OnT, the second ones is the normalized id forms used for training the geometry model-based ontology embeddings.


## Explanations

### 1. Preliminaries
For each concept in the ontology, we introduce a new atomic concept:

**Definition 1:** Given an ontology $\mathcal{O}$, for each concept $C$ that appears in $\mathcal{O}$, we denote by $N_C$ a new atomic concept that does not appear in $\mathcal{O}$, where:

- $N_C = N_D$ if and only if $C = D$ (identity preservation)
- $N_A = A$ for any atomic concept $A$ (by abuse of notation)


**Definition 2:** For an $\mathcal{EL}$ concept $C$, the concept ontology $\mathcal{O}(C)$ is inductively defined as follows:

1. If $C = A$ is an atomic concept, then $\mathcal{O}(C) = \emptyset$

2. If $C = \exists r. C_1$, then $\mathcal{O}(C) = \mathcal{O}(C_1) \cup \{N_C \equiv \exists r. N_{C_1}\}$

3. If $C = C_1 \sqcap \ldots \sqcap C_n$, then:
   $\mathcal{O}(C) = \left(\bigcup_{i=1}^n \mathcal{O}(C_i)\right) \cup \{N_C \equiv N_{C_1} \sqcap \ldots \sqcap N_{C_n}\}$

### 2. Normalization by Replacing Rules

The algorithm applies the following replacement rules exhaustively until all resulting axioms are in normal form:

1. **Equivalence Axiom Transformation**:  
   $C \equiv D \rightarrow C \sqsubseteq D, D \sqsubseteq C$

2. **Conjunctive Right-Hand Side Splitting**:  
   $C \sqsubseteq D_1 \sqcap \ldots \sqcap D_n \rightarrow C \sqsubseteq D_1, \ldots, C \sqsubseteq D_n$

3. **Complex Right-Hand Side Normalization**:  
   $C \sqsubseteq \exists r. D \rightarrow N_C \sqsubseteq \exists r. N_D, \mathcal{O}(C), \mathcal{O}(D)$

4. **Left-Hand Side Conjunction Normalization**:  
   $C_1 \sqcap C_2 \sqsubseteq A \rightarrow N_{C_1} \sqcap N_{C_2} \sqsubseteq A, \mathcal{O}(C_1), \mathcal{O}(C_2)$

5. **Left-Hand Side Existential Normalization**:  
   $\exists r. C \sqsubseteq A \rightarrow \exists r. N_C \sqsubseteq A, \mathcal{O}(C)$

### 3. Textual Description Assignment

For each new atomic concept $N_C$, we assign a textual description by verbalizing its associated original concept:

$\mathcal{V}(N_C) := \mathcal{V}(C)$

This ensures that the normalized ontology maintains human-interpretable meaning while achieving a standardized structure suitable for computational processing.

The normalization process ensures:
- All axioms are in one of the normal forms (NF1-NF4)
- Every complex concept is represented by an atomic concept
- Semantic equivalence with the original ontology is preserved
- Textual descriptions are maintained for all concepts


## Pseudocode
```python
Algorithm: OntologyNormalization
Input: Ontology O
Output: Normalized ontology O′

// Initialize data structures
O′ ← ∅
ConceptMap ← {} // Maps complex concepts to atomic representatives
AtomicConcepts ← ExtractAtomicConcepts(O)

// Pre-processing: Create concept mapping
Function OntologyForConcept(concept C):
    if IsAtomic(C) then
        return C
    if C ∉ ConceptMap then
        NC ← CreateNewAtomicConcept()
        ConceptMap[C] ← NC
        if C = ∃r.C1 then
            NC1 ← OntologyForConcept(C1)
            AddAxiom(O′, {NC ≡ ∃r.NC1})
        else if C = C1 ⊓ ... ⊓ Cn then
            for each Ci in {C1, ..., Cn} do
                NCi ← OntologyForConcept(Ci)
            AddAxiom(O′, {NC ≡ NC1 ⊓ ... ⊓ NCn})
    return ConceptMap[C]

// Main normalization
for each axiom α in O do
    if α is of form (C ≡ D) then
        AddAxioms(O′, {C ⊑ D, D ⊑ C})
    
    else if α is of form (C ⊑ D1 ⊓ ... ⊓ Dn) then
        for each Di in {D1, ..., Dn} do
            AddAxiom(O′, {C ⊑ Di})
    
    else if α is of form (C ⊑ ∃r.D) then
        NC ← OntologyForConcept(C)
        ND ← OntologyForConcept(D)
        AddAxiom(O′, {NC ⊑ ∃r.ND})
    
    else if α is of form (C1 ⊓ C2 ⊑ A) where A is atomic then
        NC1 ← OntologyForConcept(C1)
        NC2 ← OntologyForConcept(C2)
        AddAxiom(O′, {NC1 ⊓ NC2 ⊑ A})
    
    else if α is of form (∃r.C ⊑ A) where A is atomic then
        NC ← OntologyForConcept(C)
        AddAxiom(O′, {∃r.NC ⊑ A})
    
    else
        pass

// Assign textual descriptions
for each new atomic concept NC in ConceptMap do
    SetVerbalization(NC, GetVerbalization(GetOriginalConcept(NC)))

return O′
```

#This code processes a JSON file containing Human Phenotype Ontology (HPO) data to extract information about HPO terms, their relationships, and lineages.
import json
import re
from collections import defaultdict

# Extracts HPO term information (ID, label, definition, synonyms) from a node in the JSON data.
def extract_info(node):
    hpo_info = {}
    hp_id_match = re.search(r'(HP_\d+)', node['id'])
    if hp_id_match:
        hp_id = hp_id_match.group(1)
        info_dict = {'label': '', 'definition': '', 'synonyms': []}
        lbl = node.get('lbl', '')
        if lbl:
            info_dict['label'] = lbl
        if 'meta' in node and 'definition' in node['meta']:
            definition_val = node['meta']['definition'].get('val', '')
            if definition_val:
                info_dict['definition'] = definition_val
        if 'meta' in node and 'synonyms' in node['meta']:
            synonyms = [synonym.get('val', '') for synonym in node['meta']['synonyms'] if synonym.get('val', '')]
            if synonyms:
                info_dict['synonyms'] = synonyms
        if info_dict['label'] or info_dict['definition'] or info_dict['synonyms']:
            hpo_info[hp_id] = info_dict
    return hpo_info

# Reads the JSON file, processes each node to extract HPO information, and returns a dictionary of all HPO terms.
def process_json_file(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    all_hpo_info = {}
    for graph in data.get('graphs', []):
        for node in graph.get('nodes', []):
            hpo_info = extract_info(node)
            if hpo_info:
                all_hpo_info.update(hpo_info)
    return all_hpo_info

# Recursively finds all ancestral lineages for a given HPO term.
def find_lineages(term, lineage=[]):
    current_lineage = lineage + [f'{label_map.get(term, "No label")} ({term})']
    if term not in parent_map:
        return [current_lineage]
    lineages = []
    for parent in parent_map[term]:
        lineages.extend(find_lineages(parent, current_lineage))
    return lineages

# Load and process the JSON data
hpo_data = process_json_file('hp.json')

# Reads the JSON file and initializes mappings for node labels and parent-child relationships.
with open('hp.json', 'r') as file:
    data = json.load(file)
nodes = data['graphs'][0]['nodes']
edges = data['graphs'][0]['edges']
label_map = {node['id'].split('/')[-1]: node.get('lbl', 'No label') for node in nodes if 'lbl' in node}
parent_map = defaultdict(list)
for edge in edges:
    sub_id = edge['sub'].split('/')[-1]
    obj_id = edge['obj'].split('/')[-1]
    parent_map[sub_id].append(obj_id)

# Create a dictionary to store all lineages for each term
# Stores all lineages for each HPO term, excluding obsolete terms.
hpo_lineage = {}
for term in label_map.keys():
    if 'obsolete' in label_map.get(term, '').lower():
        continue
    lineages = find_lineages(term)
    hpo_lineage[term] = [' -> '.join(lineage) for lineage in lineages]

# Initialize dictionaries for tracking relationships
immediate_parents = defaultdict(set)
immediate_descendants = defaultdict(set)
all_descendants = defaultdict(set)

# Initializes and updates dictionaries to track immediate and all descendants for each HPO term.
for term, lineages in hpo_lineage.items():
    for lineage in lineages:
        terms = [t.split(' ')[-1].strip('()\n') for t in lineage.split(' -> ')]
        for i, term in enumerate(terms):
            if i < len(terms) - 1:
                immediate_parents[terms[i]].add(terms[i+1])
                immediate_descendants[terms[i+1]].add(terms[i])
            if i > 0:
                for descendant in terms[:i]:
                    all_descendants[terms[i]].add(descendant)

# Adds relationship counts (unique parents, immediate descendants, total descendants) to each HPO term.
for hpo_id in set(immediate_parents.keys()).union(immediate_descendants.keys()).union(all_descendants.keys()):
    if hpo_id not in hpo_data:
        hpo_data[hpo_id] = {"Description": "No description available"}
    hpo_data[hpo_id]["Unique_Parent_Count"] = len(immediate_parents[hpo_id])
    hpo_data[hpo_id]["Immediate_Descendant_Count"] = len(immediate_descendants[hpo_id])
    hpo_data[hpo_id]["Total_Descendant_Count"] = len(all_descendants[hpo_id])

# Adds sorted lineage information to each HPO term.
lineages_by_term = defaultdict(list)
for term, lineages in hpo_lineage.items():
    lineages_by_term[term].extend(lineages)

for hpo_id, lineages in lineages_by_term.items():
    sorted_lineages = sorted(lineages, key=lambda x: len(x.split(' -> ')))
    if hpo_id in hpo_data:
        hpo_data[hpo_id]["lineage"] = sorted_lineages
    else:
        hpo_data[hpo_id] = {
            "Description": "No description available",
            "lineage": sorted_lineages
        }

# Saves the updated HPO data with lineage information to a new JSON file.
with open('hpo_data_with_lineage.json', 'w') as file:
    json.dump(hpo_data, file, indent=4)
print("The final HPO data with lineage information has been saved to 'hpo_data_with_lineage.json'.")
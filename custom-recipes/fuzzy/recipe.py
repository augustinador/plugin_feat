

import dataiku
from dataiku.customrecipe import *


input_A_names = get_input_names_for_role('input_A_role')[0]


# For outputs, the process is the same:
output_A_names = get_output_names_for_role('main_output')[0]


# The configuration consists of 

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
COL_BLOCK = get_recipe_config()['COL_BLOCK']
COL_TO_COMPARE = get_recipe_config()['COL_TO_COMPARE']
print("here")
print(COL_TO_COMPARE)
UNIQUE = get_recipe_config()['UNIQUE']


THRESHOLD = int(get_recipe_config()['threshold'])




# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import recordlinkage
import pandas

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
import recordlinkage
from recordlinkage.datasets import load_febrl1

UNIQUE = "_row_number"
COL_BLOCK = "Name_1_2_combined"
COL_TO_COMPARE = ["Name_1", "Street_1", "House_Number_1"]
THRESHOLD = 0.7

dataset_X01_BusinessPartner_filtered = dataiku.Dataset(input_A_names)
dfA = dataset_X01_BusinessPartner_filtered.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Indexation step
indexer = recordlinkage.Index()

indexer = recordlinkage.SortedNeighbourhoodIndex(
        COL_BLOCK, window=9)



#indexer.block(left_on='Name_1_2_combined')


candidate_links = indexer.index(dfA)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Comparison step
compare_cl = recordlinkage.Compare()

for col_name in COL_TO_COMPARE:
    compare_cl.string(col_name, col_name, method='damerau_levenshtein', threshold=THRESHOLD)

print('does it work?')
#compare_cl.string('Matchcode_Term_1', 'Matchcode_Term_1', method='damerau_levenshtein', threshold=0.85)


features = compare_cl.compute(candidate_links, dfA)

# Classification step
features = features[features.sum(axis=1) >= len(COL_TO_COMPARE)]
features = features.reset_index()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
COL_TO_COMPARE.append(COL_BLOCK)
COL_TO_COMPARE.append(UNIQUE)

tmp1 = features.merge(dfA[COL_TO_COMPARE] , how='inner', left_index=False, right_index=True, left_on="level_0")
tmp2= features[["level_0", "level_1"]].merge(dfA[COL_TO_COMPARE], how='inner', left_index=False, right_index=True, left_on="level_1")

features = tmp1.merge(tmp2, how='inner', left_index=True, right_index=True).drop([ "level_0_x", "level_1_x", "level_0_y", "level_1_y"], axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
plugin_test = dataiku.Dataset(output_A_names)
plugin_test.write_with_schema(features)
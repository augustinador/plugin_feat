# Code for custom code recipe fuzzy (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_A_names = get_input_names_for_role('input_A_role')
# The dataset objects themselves can then be created like this:
input_A_datasets = [dataiku.Dataset(name) for name in input_A_names]

# For outputs, the process is the same:
output_A_names = get_output_names_for_role('main_output')
output_A_datasets = [dataiku.Dataset(name) for name in output_A_names]


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
my_variable = get_recipe_config()['parameter_name']

# For optional parameters, you should provide a default value in case the parameter is not present:
my_variable = get_recipe_config().get('parameter_name', None)

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

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

COL_BLOCK = "Name_1_2_combined"
COL_TO_COMPARE = ["Name_1", "Street_1", "House_Number_1"]
THRESHOLD = 0.7

dataset_X01_BusinessPartner_filtered = dataiku.Dataset("out_emb_sorted")
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


#compare_cl.string('Matchcode_Term_1', 'Matchcode_Term_1', method='damerau_levenshtein', threshold=0.85)


features = compare_cl.compute(candidate_links, dfA)

# Classification step
features = features[features.sum(axis=1) >= len(COL_TO_COMPARE)]
features = features.reset_index()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
COL_TO_COMPARE.append(COL_BLOCK)
COL_TO_COMPARE.append("_row_number")

tmp1 = features.merge(dfA[COL_TO_COMPARE] , how='inner', left_index=False, right_index=True, left_on="level_0")
tmp2= features[["level_0", "level_1"]].merge(dfA[COL_TO_COMPARE], how='inner', left_index=False, right_index=True, left_on="level_1")

features = tmp1.merge(tmp2, how='inner', left_index=True, right_index=True).drop([ "level_0_x", "level_1_x", "level_0_y", "level_1_y"], axis=1)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
plugin_test = dataiku.Dataset("plugin_test")
plugin_test.write_with_schema(features)
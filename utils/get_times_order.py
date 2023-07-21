"""
Utility script to get the order of sets/parameters as given in the TIMES GAMS source files.
Run a times model, and point this script to the SCENARIO.gdx file that it generates.
This script extracts the order of sets/parameters from the gdx file and prints the order
of tables in times_mapping.txt that is consistent with the gdx file.

Dependencies:
  GAMS
  GAMS Python API
  gdx-pandas package

Installation:
pip install 'gams[all]' --find-links /Library/Frameworks/GAMS.framework/Versions/Current/Resources/api/python/bdist
pip install git+https://github.com/NREL/gdx-pandas.git@main

"""

import gdxpds

mapping_tables = set(
    map(lambda l: l.split("[")[0], open("times_mapping.txt").readlines())
)
with gdxpds.gdx.GdxFile() as f:
    f.read("benchmarks/dd/DemoS_001-all/SCENARIO.gdx")
    gams_order = [t for t in f.keys() if t in mapping_tables]

print(
    f"Missing {len(mapping_tables)-len(gams_order)} tables: {mapping_tables - set(gams_order)}"
)
print(gams_order)


# This was my failed attempt to write a small LARK parser to directly parse
# GAMS source files to extract the order of sets and parameters:

failed_small_grammar = r"""
start: statement+

?statement: (symbol_definition
	| other_statement
	| DOLLAR_CONTROL_OPT
	| COMMENT )

DOLLAR_CONTROL_OPT: /\$[^\n]+/
COMMENT : /\*[^\n]+/
COMMENT_BLOCK: "$ontext" /(\S|\s)*?/ "$offtext"
%import common.WS
%import common.NUMBER
%import common.NEWLINE -> _NL
%ignore COMMENT_BLOCK
%ignore WS

symbol_definition: _SET 		definition+ 		_END -> set_list
				| _PARAMETER 	definition+ 		_END -> parameter_list

definition: identifier _ANYTHING_TILL_SLASH _SLASH _SLASH [","]
	| identifier _ANYTHING_TILL_SLASH _SLASH _ANYTHING_TILL_SLASH _SLASH [","]

other_statement: _NOT_SET_PARAM _ANYTHING_TILL_END _END

?identifier 	: WORD_IDENTIFIER

_SET: "sets" | "Sets" | "Set" | "SET" | "set" | "SETS"
_PARAMETER: "Parameter" | "PARAMETER" | "parameter" | "PARAMETERS"

_NOT_SET_PARAM: /(?!SET)[a-zA-Z]/

WORD_IDENTIFIER : /[a-zA-Z][\w-]*/
_ANYTHING_TILL_END: /[^;]+/
_ANYTHING_TILL_SLASH: /[^\/]+/

_SLASH: "/"
_END: ";"
"""

"""
import gams_parser as gp

ast = gp.GamsParser("TIMES_model/initsys.mod").parse()
# ast = gp.GamsParser("initmty.vda").parse()
print(ast.pretty())

mapping_tables = set(map(lambda l: l.split('[')[0], open('times_mapping.txt').readlines()))
gams_order = []
def get_tables_from_file(f):
    ast = gp.GamsParser(f).parse()
    for node in ast.iter_subtrees_topdown():
        if node.data in ['set_list', 'parameter_list']:
            for defn in node.children:
                gams_order.append(defn.children[0].children[0].children[0].value)
get_tables_from_file("TIMES_model/initmty.mod")
gams_order = [t for t in gams_order if t in mapping_tables]
print(f"Missing {len(mapping_tables)-len(gams_order)} tables: {mapping_tables - set(gams_order)}")
"""

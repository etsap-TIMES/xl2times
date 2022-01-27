from openpyxl import load_workbook
from openpyxl.worksheet.cell_range import CellRange
from pandas.core.frame import DataFrame
import pandas as pd
from dataclasses import dataclass, replace
from typing import Dict, List, Set, Union
from more_itertools import locate
from itertools import groupby
import numpy
import re
import os
from multiprocessing import Pool
from math import log10, floor
import time

@dataclass
class EmbeddedXlTable:
    tag: str
    sheetname: str
    range: str
    tag_indent: int
    filename: str
    dataframe: DataFrame


@dataclass
class TimesXlMap:
    times_name: str
    times_cols: List[str]
    xl_name: str
    xl_cols: List[str]
    col_map: Dict[str, str]


def extract_tables(filename: str) -> List[EmbeddedXlTable]:
    print(f"Loading {filename} ...")
    workbook = load_workbook(filename=filename, data_only=True)

    tables = []
    for sheet in workbook.worksheets:
        df = pd.DataFrame(sheet.values)
        for colname in df.columns:
            for row, value in df[colname].astype('str').items():
                if value.startswith('~'):
                    col = df.columns.get_loc(colname)
                    tables.append(extract_table(row, col, df, sheet.title, filename))
    
    return tables


def extract_table(tag_row: int, tag_col: int, df: DataFrame, sheetname: str, filename: str) -> EmbeddedXlTable:
    header_row = tag_row + 1

    start_col = tag_col
    while not cell_is_empty(df.iloc[header_row, start_col - 1]):
        start_col -= 1

    end_col = tag_col
    while end_col < df.shape[1] and not cell_is_empty(df.iloc[header_row, end_col]):
        end_col += 1

    end_row = header_row
    while end_row < df.shape[0] and not are_cells_all_empty(df, end_row, start_col, end_col):
        end_row += 1

    range = str(CellRange(min_col=start_col, min_row=header_row, max_col=end_col, max_row=end_row))

    if end_row - header_row == 1 and end_col - start_col == 1:
        # Interpret single cell tables as a single data item with a column name VALUE
        table_df = DataFrame(df.iloc[header_row, start_col:end_col])
        table_df.columns = ["VALUE"]
        table_df.reset_index(drop=True, inplace=True)
    else:
        table_df = DataFrame(df.iloc[header_row+1:end_row, start_col:end_col])
        table_df.columns = df.iloc[header_row, start_col:end_col]
        table_df.reset_index(drop=True, inplace=True)

    return EmbeddedXlTable(
        filename = filename,
        sheetname = sheetname,
        range = range,
        tag = df.iloc[tag_row, tag_col],
        tag_indent = tag_col - start_col,
        dataframe = table_df
    )


def are_cells_all_empty(df, row: int, start_col: int, end_col: int) -> bool:
    for col in range(start_col, end_col):
        if not cell_is_empty(df.iloc[row, col]):
            return False
    return True

def cell_is_empty(value) -> bool:
    return value is None or (isinstance(value,numpy.float64) and numpy.isnan(value))

def remove_comment_rows(table: EmbeddedXlTable) -> EmbeddedXlTable:
    comment_rows = list(locate(table.dataframe.iloc[:, 0], lambda cell: isinstance(cell, str) and cell.startswith('*')))
    df = table.dataframe.drop(index=comment_rows)
    df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df)


def remove_comment_cols(table: EmbeddedXlTable) -> EmbeddedXlTable:
    comment_cols = list(locate(table.dataframe.columns, lambda cell: isinstance(cell, str) and cell.startswith('*')))
    df = table.dataframe.drop(columns=[table.dataframe.columns[i] for i in comment_cols])
    df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df, tag_indent=table.tag_indent - len(comment_cols))

def remove_tables_with_formulas(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    def is_formula(s):
        return isinstance(s,str) and len(s)>0 and s[0] == '='

    def has_formulas(table):
        has = table.dataframe.applymap(is_formula).any(axis=None)
        if has:
            print(f"WARNING: Excluding table {table.tag} because it has formulas")
        return has
    
    return [table for table in tables if not has_formulas(table)]

def merge_tables(tables: List[EmbeddedXlTable]) -> Dict[str, DataFrame]:
    """Merge tables of the same types"""
    result = {}
    for key, value in groupby(sorted(tables, key=lambda t: t.tag), lambda t: t.tag):
        group = list(value)
        if not all(set(t.dataframe.columns) == set(group[0].dataframe.columns) for t in group):
            print(f"WARNING: Cannot merge tables with tag {key} as their columns are not identical")
        else:
            result[key] = pd.concat([table.dataframe for table in group])
    return result

def apply_composite_tag(table: EmbeddedXlTable) -> EmbeddedXlTable:
    """Process composite tags e.g. ~FI_T: COM_PKRSV"""
    if ':' in table.tag:
        (newtag, varname) = table.tag.split(': ')
        df = table.dataframe.copy()
        df["Attribute"].fillna(varname, inplace=True)
        return replace(table, tag=newtag, dataframe=df)
    else:
        return table

def explode(df, data_columns):
    data = df[data_columns].values.tolist()
    other_columns = [colname for colname in df.columns.values if colname not in data_columns]
    df = df[other_columns]
    value_column = 'VALUE'
    df = df.assign(VALUE = data)
    nrows = df.shape[0]
    df = df.explode(value_column)
    if isinstance(data_columns,numpy.ndarray):
        data_columns = data_columns.astype('str').tolist()
    names = pd.Series(data_columns * nrows, index=df.index)
    # Remove rows with no VALUE
    filter = df[value_column].notna()
    df = df[filter]
    names = names[filter]
    return df, names

def process_tech_tables(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    legal_values = {
        "LimType" : {"LO", "UP", "FX"},
        "TimeSlice" : {"ANNUAL", "SEASON", "DAYNITE"},
        "Comm-OUT" : set(merge_columns(tables, "~FI_Comm", "CommName"))
    }

    def get_colname(value):
        if value.isdigit():
            return 'Year',int(value)
        for name, values in legal_values.items():
            if value in values:
                return name,value

    def process_tech_table(table: EmbeddedXlTable) -> EmbeddedXlTable:
        if not table.tag.startswith('~FI_T'):
            return table
        df = table.dataframe
        mapping = {'YEAR':'Year','CommName':'Comm-IN'}
        df = df.rename(columns=mapping)
        nrows = df.shape[0]
        if ('Comm-IN' in df.columns) and ('Comm-OUT' in df.columns):
            kwargs = {'TOP-IN' : ['IN'] * nrows, 'TOP-OUT' : ['OUT'] * nrows}
            df = df.assign(**kwargs)
        data_columns = df.columns.values[(table.tag_indent+1):df.shape[1]]
        # Remove any TechDesc column
        if 'TechDesc' in df.columns:
            df.drop('TechDesc', axis=1, inplace=True)
        index_columns = ["TechName", "Comm-IN", "Comm-OUT", "Comm-OUT-A", "Year", "TimeSlice", "LimType", "Region", "Curr", "Other_Indexes", "Attribute"]
        for colname in index_columns:
            if colname not in df.columns:
                df[colname] = [None] * nrows
        table = replace(table, dataframe=df)
        table = apply_composite_tag(table)
        df = table.dataframe
        df, attribute_suffix = explode(df, data_columns)
        
        # Append the data column name to the Attribute column
        attribute = 'Attribute'
        if nrows > 0:
            i = df[attribute].notna()
            df.loc[i,attribute] = df.loc[i,attribute] + '~' + attribute_suffix[i]
            i = df[attribute].isna()
            df.loc[i,attribute] = attribute_suffix[i]

        # Handle Attribute containing tilde, such as 'STOCK~2030'
        for attr in df[attribute].unique():
            if '~' in attr:
                i = df[attribute] == attr
                parts = attr.split('~')
                for value in parts[1:]:
                    colname, typed_value = get_colname(value)
                    if colname is None:
                        raise ValueError(f'Unrecognized value {value}')
                    df.loc[i,colname] = typed_value
                df.loc[i,attribute] = parts[0]

        # Handle Other_Indexes
        other = 'Other_Indexes'
        for attr in df[attribute].unique():
            if attr == 'FLO_EMIS':
                i = df[attribute] == attr
                df.loc[i & df[other].isna(),other] = "ACT"
            elif attr == 'EFF':
                i = df[attribute] == attr
                df.loc[i,"Comm-IN"] = "ACT"
                df.loc[i,attribute] = "CEFF"
            elif attr == 'OUTPUT':
                i = df[attribute] == attr
                df.loc[i,"Comm-IN"] = df.loc[i,"Comm-OUT-A"]
                df.loc[i,attribute] = "CEFF"
            elif attr == 'END':
                i = df[attribute] == attr
                df.loc[i,"Year"] = df.loc[i,"VALUE"].astype('int') + 1
                df.loc[i,other] = "EOH"
                df.loc[i,attribute] = "PRC_NOFF"
            elif attr == 'TOP-IN':
                i = df[attribute] == attr
                df.loc[i,other] = df.loc[i,"Comm-IN"]
                df.loc[i,attribute] = "IO"
            elif attr == 'TOP-OUT':
                i = df[attribute] == attr
                df.loc[i,other] = df.loc[i,"Comm-OUT"]
                df.loc[i,attribute] = "IO"
        filter = ~((df[attribute] == "IO") & df[other].isna())
        df = df[filter]

        if len(df.columns) != 12:
            raise ValueError(f'len(df.columns) = {len(df.columns)}')

        return replace(table, dataframe=df)

    return [process_tech_table(t) for t in tables]


def merge_columns(tables: List[EmbeddedXlTable], tag: str, colname: str):
    columns = [table.dataframe[colname].values for table in tables if table.tag == tag]
    return numpy.concatenate(columns)

def fill_in_missing_values(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    regions = merge_columns(tables, "~BookRegions_Map", 'Region')
    start_year = merge_columns(tables, "~StartYear", 'VALUE')[0]
    currency = merge_columns(tables, "~Currencies", 'Currency')[0]

    for table in tables:
        df = table.dataframe.copy()
        for colname in df.columns:
            # TODO make this more declarative
            if colname == "Csets" or colname == "TechName":
                missing_value_inherit(df[colname])
            elif colname == "LimType" and table.tag == "~FI_Comm" and False:
                isna = df[colname].isna()
                ismat = df['Csets'] == "MAT"
                df.loc[isna & ismat,colname] = "FX"
                df.loc[isna & ~ismat,colname] = "LO"
            elif colname == "LimType" and table.tag == "~FI_T":
                isna = df[colname].isna()
                islo = df['Attribute'].isin({"BS_STIME","GR_VARGEN","RCAP_BND"})
                isfx = df['Attribute'].isin({"ACT_LOSPL","FLO_SHAR","MARKAL-REH","NCAP_CHPR","VA_Attrib_C","VA_Attrib_T","VA_Attrib_TC"})
                df.loc[isna & islo,colname] = "LO"
                df.loc[isna & isfx,colname] = "FX"
                df.loc[isna & ~islo & ~isfx,colname] = "UP"
            elif colname == "TimeSlice" or colname == "Tslvl": # or colname == "CTSLvl" or colname == "PeakTS":
                df[colname].fillna("ANNUAL", inplace=True) # ACT_CSTUP should use DAYNITE
            elif colname == "Region":
                df[colname].fillna(','.join(regions), inplace=True)
            elif colname == "Year":
                df[colname].fillna(start_year, inplace=True)
            elif colname == "Curr":
                df[colname].fillna(currency, inplace=True)
        result.append(replace(table, dataframe=df))
    return result


def missing_value_inherit(series: pd.Series):
    last = None
    for index, value in series.items():
        if value == None:
            series[index] = last
        else:
            last = value


def expand_rows(table: EmbeddedXlTable) -> EmbeddedXlTable:
    """Expand out certain columns with entries containing commas"""

    # TODO pull this out
    cols = ["Csets", "Region", "PeakTS", "TS_MAP", "TS_GROUP"]

    def has_comma(s):
        return isinstance(s,str) and ',' in s

    def split_by_commas(s):
        if has_comma(s):
            return s.split(',')
        else:
            return s

    df = table.dataframe.copy()
    c = df.applymap(has_comma)
    columns_with_commas = [colname for colname in c.columns.values if c[colname].any()]
    if len(columns_with_commas) > 0:
        # Transform comma-separated strings into lists
        df = df.applymap(split_by_commas)
        for colname in columns_with_commas:
            # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.explode.html#pandas.DataFrame.explode
            df = df.explode(colname)
    return replace(table, dataframe=df)


def remove_invalid_values(table: EmbeddedXlTable) -> EmbeddedXlTable:
    # TODO pull this out
    constraints = {
        "Csets": { "NRG", "MAT", "DEM", "ENV", "FIN" }
    }

    df = table.dataframe.copy()
    is_valid = [df[colname].isin(values) for colname, values in constraints.items() if colname in df.columns]
    # This loop should be avoidable by combining the is_valid Series
    for v in is_valid:
        df = df[v]
    df.reset_index(drop=True, inplace=True)
    return replace(table, dataframe=df)


def read_mappings(filename: str) -> List[TimesXlMap]:
    mappings = []
    with open(filename) as file:
        while True:
            line = file.readline().rstrip()
            if line == '':
                break
            (times, xl) = line.split(" = ")
            (times_name, times_cols_str) = list(filter(None, re.split('\[|\]', times)))
            (xl_name, xl_cols_str) = list(filter(None, re.split('\(|\)', xl)))
            times_cols = times_cols_str.split(',')
            xl_cols = xl_cols_str.split(',')
            col_map = {}
            for index, value in enumerate(xl_cols):
                col_map[value] = times_cols[index]
            entry = TimesXlMap(times_name=times_name, times_cols=times_cols, xl_name=xl_name, xl_cols=xl_cols, col_map=col_map)

            # TODO remove: Filter out mappings that are not yet finished
            if entry.xl_name != "~TODO" and not any(c.startswith("TODO") for c in entry.xl_cols):
                mappings.append(entry)
            else:
                print(f"WARNING: Dropping mapping that is not yet complete {line}")
    return mappings


def process_time_periods(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    start_year = next(filter(lambda t : t.tag == "~StartYear", tables)).dataframe['VALUE'].values

    def process_time_periods_table(table: EmbeddedXlTable):
        if table.tag != "~TimePeriods":
            return table

        df = table.dataframe.copy()
        df.rename(columns={'Pdef-1': 'D'}, inplace=True)

        df = df.assign(B=pd.Series([None] * df.shape[0]).values)
        df.loc[0, 'B'] = start_year
        for i in range(1, df.shape[0]):
            df.loc[i, 'B'] = df.loc[i-1, 'B'] + df.loc[i-1, 'D']

        df['E'] = df.B + df.D - 1
        df['M'] = df.B + ((df.D - 1) // 2)
        df['Year'] = df.M

        return replace(table, dataframe=df)

    return [process_time_periods_table(table) for table in tables]

def process_comemi(table: EmbeddedXlTable) -> EmbeddedXlTable:
    if table.tag != "~COMEMI":
        return table

    df = table.dataframe.copy()
    index_columns = ["Region","Year","CommName"]
    data_columns = [colname for colname in df.columns.values if colname not in index_columns]
    df, names = explode(df, data_columns)
    df.rename(columns={'VALUE': 'EMCB'}, inplace=True)
    df = df.assign(Other_Indexes=names)
    nrows = df.shape[0]
    for colname in index_columns:
        if colname not in df.columns:
            df[colname] = [None] * nrows
    return replace(table, dataframe=df)

def process_time_slices(tables: List[EmbeddedXlTable]) -> List[EmbeddedXlTable]:
    result = []
    regions = ",".join(next(filter(lambda t : t.tag == "~BookRegions_Map", tables)).dataframe['Region'].values)

    for table in tables:
        if table.tag != "~TimeSlices":
            result.append(table)
        else:
            # Has to generate another table to avoid tables with empty entries
            seasons = table.dataframe['Season'].values
            day_night = table.dataframe['DayNite'].values
            df = pd.DataFrame(
                list([
                    regions,
                    s,
                    ",".join((s + d) for d in day_night)
                    ] for s in seasons),
                columns=['Region', 'Season', 'TS_MAP'])
            result.append(replace(table, dataframe=df))
            
            combo = list((s + d) for s in seasons for d in day_night)
            data = {
                'Region': regions,
                'TSLVL': ["ANNUAL", "DAYNITE", "SEASON"],
                'TS_GROUP': ["ANNUAL", ",".join(combo), ",".join(seasons)]
                }
            result.append(replace(table, tag="~TimeSlicesGroup", dataframe=DataFrame(data)))
    return result


def produce_times_tables(input: Dict[str, DataFrame], mappings: List[TimesXlMap]) -> Dict[str, DataFrame]:
    print(f"produce_times_tables: {len(input)} tables incoming, {sum(len(value) for (_, value) in input.items())} rows")
    result = {}
    for mapping in mappings:
        if not mapping.xl_name in input:
            print(f"WARNING: Cannot produce table {mapping.times_name} because input table {mapping.xl_name} does not exist")
        else:
            df = input[mapping.xl_name]
            if 'Attribute' in df.columns:
                colname = mapping.xl_cols[-1]
                i = df['Attribute'].isin({colname,mapping.times_name})
                df = df.loc[i,:]
                if colname not in df.columns:
                    df = df.rename(columns={'VALUE':colname})
            if not all(c in df.columns for c in mapping.xl_cols):
                missing = set(mapping.xl_cols) - set(df.columns)
                print(f"WARNING: Cannot produce table {mapping.times_name} because input table {mapping.xl_name} does not contain the required columns - {', '.join(missing)}")
            else:
                cols_to_drop = [x for x in df.columns if not x in mapping.xl_cols]
                df = df.copy()
                df.drop(columns=cols_to_drop, inplace=True)
                df.drop_duplicates(inplace=True)
                df.reset_index(drop=True, inplace=True)
                df.rename(columns=mapping.col_map, inplace=True)
                i = df[mapping.times_cols[-1]].notna()
                df = df.loc[i,mapping.times_cols]
                df = df.applymap(lambda cell: cell if not isinstance(cell, float) else round_sig(cell, 15))
                result[mapping.times_name] = df

    return result


def convert_xl_to_times(dir: str, input_files: List[str], mappings: List[TimesXlMap]) -> Dict[str, DataFrame]:
    raw_tables = []
    filenames = [os.path.join(dir, filename) for filename in input_files]
    with Pool(len(filenames)) as p:
        for result in p.map(extract_tables, filenames):
            raw_tables.extend(result)
    print(f"Extracted {len(raw_tables)} tables, {sum(table.dataframe.shape[0] for table in raw_tables)} rows")

    # For debugging
    os.makedirs("output", exist_ok=True)
    with open(r"output/raw_tables.txt", "w") as text_file:
        for t in raw_tables:
            text_file.write(f"tag: {t.tag}\n")
            text_file.write(f"sheetname: {t.sheetname}\n")
            text_file.write(f"range: {t.range}\n")
            text_file.write(f"filename: {t.filename}\n")
            text_file.write(t.dataframe.to_csv(index=False, line_terminator='\n'))
            text_file.write("\n" * 2)

    transforms = [
        lambda tables: [remove_comment_rows(t) for t in tables],
        lambda tables: [remove_comment_cols(t) for t in tables],
        remove_tables_with_formulas, # slow

        process_tech_tables, # slow
        lambda tables: [process_comemi(t) for t in tables],
        fill_in_missing_values, # slow
        process_time_slices,
        lambda tables: [expand_rows(t) for t in tables], # slow
        lambda tables: [remove_invalid_values(t) for t in tables],
        process_time_periods,

        merge_tables,

        lambda tables: produce_times_tables(tables, mappings)
    ]

    results = []
    input = raw_tables
    for transform in transforms:
        start_time = time.time()
        output = transform(input)
        end_time = time.time()
        print(f"transform took {end_time-start_time} seconds")
        results.append(output)
        input = output

    print(f"Conversion complete, {len(output)} tables produced, {sum(df.shape[0] for tablename, df in output.items())} rows")

    return output


def write_csv_tables(tables: Dict[str, DataFrame], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    for item in os.listdir(output_dir):
        if item.endswith(".csv"):
            os.remove(os.path.join(output_dir, item))
    for tablename, df in tables.items():
        df.to_csv(os.path.join(output_dir, tablename + ".csv"), index=False)


def read_csv_tables(input_dir: str) -> Dict[str, DataFrame]:
    result = {}
    for filename in os.listdir(input_dir):
        result[filename.split('.')[0]] = pd.read_csv(os.path.join(input_dir, filename))
    return result


def compare(data: Dict[str, DataFrame], ground_truth: Dict[str, DataFrame]):
    print(f"Ground truth contains {len(ground_truth)} tables, {sum(df.shape[0] for tablename, df in ground_truth.items())} rows")

    missing = set(ground_truth.keys()) - set(data.keys())
    missing_str = ",".join(sorted(missing))
    if len(missing) > 0:
        print(f"WARNING: Missing {len(missing)} tables: {missing_str}")

    for table_name, gt_table in ground_truth.items():
        if table_name in data:
            data_table = data[table_name]
            if list(gt_table.columns) != list(data[table_name].columns):
                print(f"WARNING: Table {table_name} header incorrect, was {data_table.columns.values}, should be {gt_table.columns.values}")
            else:
                gt_rows = set(tuple(i) for i in gt_table.values.tolist())
                data_rows = set(tuple(i) for i in data_table.values.tolist())
                extra = data_rows - gt_rows
                if len(extra) > 0:
                    print(f"WARNING: Table {table_name} contains {len(extra)} rows out of {data_table.shape[0]} that are not present in the ground truth")


def round_sig(x, sig_figs):
    return round(x, -int(floor(log10(abs(x)))) + sig_figs - 1)

        
if __name__ == "__main__":
    mappings = read_mappings("times_mapping.txt")

    xl_files_dir = "input"
    input_files = ["SysSettings.xlsx", "VT_UK_RES.xlsx", "VT_UK_ELC.xlsx", "VT_UK_IND.xlsx", "VT_UK_AGR.xlsx"]

    tables = convert_xl_to_times(xl_files_dir, input_files, mappings)

    write_csv_tables(tables, "output")
    
    ground_truth = read_csv_tables("ground_truth")
    compare(tables, ground_truth)
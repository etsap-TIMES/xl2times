## High level TODO *(rough priority/dependency order)*</sub>

- Generate all output rows that are in ground truth (~40% complete currently)
  - I estimate this task is ~80% of the remaining work
  - See CI for latest status
  - See https://github.com/microsoft/times-excel-reader/wiki/State-of-Progress
- Decide how we should manage work list and assignments - TODO file, Wiki, Issues, Milestones
- Write an overview of how code currently works to help get new people started
- Go over old TODO lists below and work out what is still relevant
- Code formatting
- Integrate type checker into CI (mypy?)
- Decide at what point to reach out to TIMES community and announce project (and ask for help?)
- Decide what to do about ~TFM_FILL. These tables look like the Excel files are meant to be updated by the tool
- Create a glossary?
- Remove additional rows in output that are not in ground truth
- Test on other data sets (using CI?)
- Split into scenarios (we are currently just lumping all the data together into one)
- Sort out row ordering (currently ignored)
- Generate (or convert to) DD files
- Measure difference between our generated DD files and ground truth
- Get difference between generated DD files and ground truth to zero
- Code tidying


## OLD High level TODO list

- [ ] FI_Comm (may be done except for a check)
- [ ] FI_Process
- [ ] PRC_ACTUNT is wrong
- [ ] PRC_MAP is wrong
- [ ] PRC_TSL is wrong (dd conflicts with input tables)
- [ ] PRC_VINT is wrong
- [ ] NCAP_PKCNT is wrong (bug in input tables?)
- [ ] NRG_TMAP is wrong
- [ ] COM_GRP is wrong
- [ ] COM_TSL is wrong (includes commodities that are never used)
- [ ] DATAYEAR is wrong
- [ ] ACT_EFF numbers are wrong (bug in VEDA?)
- [ ] PRC_RESID is wrong (bug in VEDA? VEDA converts 22.4486668069665 into 22.4486668069666)
- [ ] VDA_EMCB numbers are wrong
- [ ] VDA_FLOP is missing COM_GRP
- [ ] FLO_SHAR is missing COM_GRP
- [ ] Print unused Attributes


## OLD Mix of doc and detailed TODO list

Steps to standardize a commodity table (~FI\_COMM)

- [x] Extract table by finding tag and blank rows and columns
- [x] Remove comment rows
- [x] Remove table tag row (whilst remembering metadata like its location)
- [x] Convert to data frame
- [x] Remove comment columns and reduce tag\_indent accordingly (see last table in VT\_UK\_IND[IIS])
- [x] Concatenate all commodity tables by row
- [x] Apply column-specific missing value handling
  - [x] Csets: declarations are inherited until the next one is encountered
  - [x] Region: By default, it is applied to all regions of the model when not specified
    - [x] Get hold of regions from  ~BookRegions\_Map, can use comma-separated list here
- [x] LimType: When not specified, the default is LO for all but MAT commodities (Csets column) with a default of FX
- [x] CTSLvl: When not specified, the default is ANNUAL
- [x] PeakTS: If not specified the default is ANNUAL
- [x] Duplicate rows with comma-separated values (Csets, Region, PeakTS), each row getting only one of the values.
- [ ] After above, check there are no comma-separated values
- [x] Remove rows with disallowed values
  - [x] Csets must be one of: NRG (energy), MAT (material), DEM (demand service), ENV (emissions) and FIN (financial)
- [x] Split columns into separate tables, permute/rename/drop columns to match TIMES (using times\_mapping.txt)
- [x] Mark each table as set or parameter (based having a column named "VALUE")
- [ ] Put each table into the correct dd file (syssettings.dd, base.dd, etc).  This is based on the name of the Excel file.

<br/>

Notes:

For now, ignore all validity constraints (assume input is valid)

Try enumerating all possible "missing value handling rules" and mark all columns with one of them

There are processes in the Excel files that are not mentioned in the dd files

Row ordering must be preserved by these transformations

<br/>

Extra steps for process tables:

- [ ] Remove rows for unused processes

Missing-value handling of process tables (~FI\_PROCESS)

- [ ] Sets: Inherit values from previous row if missing
- [ ] Region: all regions if missing
- [ ] Tslvl: default value depends on the value of Sets column
- [ ] PrimaryCG: if missing, use the output commodity
  - [ ] Can only be done after the technology tables are available
  - [ ] Could append the output commodity column first
- [ ] Others have constant/fixed default values

<br/>

Special handling for flexible import tables (~FI\_T)

- [ ] Create a column named "Attribute" if it does not yet exist
  - [ ] Same for Year, TimeSlice, LimType, Region, Currency
- [x] If the table tag has a table-level declaration (e.g., ~FI\_T:DEMAND):
  - [x] Set every missing value for the Attribute column to the tag value
  - [x] Example: ~FI\_T: COM\_PKRSV
    - [x] Set every missing value of Attribute to "COM\_PKRSV"
- [ ] Add two new data columns:
  - [ ] TOP-IN whose value is "IN"
  - [ ] TOP-OUT whose value is "OUT"
- [ ] Use the column position of the table tag to identify the data columns (columns to the right of the tag) for the next step
- [ ] For each row, collect all data columns and transpose them to get a single column with multiple rows.  The new column is named "VALUE".
  - [ ] Copy the existing non-data columns across the new rows
  - [ ] The original column names are now Attribute values.  If an Attribute value already exists, append the column name to it, with a tilde in the middle.  Example: Attribute was DEMAND, column name was "2010", new Attribute is "DEMAND~2010".
- [ ] Concatenate all technology tables by row (they should have the same columns at this point)
- [ ] If any Attribute contains tilde (e.g., STOCK~2030):
  - [ ] Determine what indices are being given (out of the six above plus Comm-OUT). A number is always Year.  A non-number must be matched against the possible values for each index.
  - [ ] If the table column for any index does not yet exist, create it
  - [ ] For each row, move the index string from the Attribute to the index column
  - [ ] Example: Attribute is STOCK~2030
    - [ ] The index is Year
    - [ ] New Attribute is "STOCK", Year=2030
  - [ ] Example: Attribute is FLOSHAR~UP
    - [ ] The index is LimType
    - [ ] New Attribute is "FLOSHAR", LimType="UP"
  - [ ] Example: Attribute is FLOSHAR~UP~0
    - [ ] The indices are LimType and Year
    - [ ] New Attribute is "FLOSHAR", LimType="UP", Year="0" (The data for year “0” are interpolation/extrapolation options)
- [ ] Missing-value handling after the above is done:
  - [ ] Region: all regions if missing
  - [ ] Year: the base year if missing
  - [ ] TechName: inherited from above
  - [ ] LimType: TODO (dd files have limits, called BD, but not sure how this is determined)
- [ ] If Attribute="FLOSHAR":
  - [ ] If Other\_Indexes is missing, set it to the value in COM\_GMAP(Comm-IN)
- [ ] If Attribute="FLO\_EMIS":
  - [ ] If Other\_Indexes is missing, set it to "ACT"
- [ ] If Attribute="EFF":
  - [ ] Change Comm-IN to "ACT"
  - [ ] Change Attribute to "CEFF"
- [ ] If Attribute="OUTPUT":
  - [ ] Change Comm-IN to "Comm-OUT-A"
  - [ ] Change Attribute to "CEFF"
- [ ] If Attribute="END":
  - [ ] Change Year to the contents of VALUE + 1
  - [ ] Change Other\_Indexes to "EOH"
  - [ ] Change Attribute to "PRC\_NOFF"
- [ ] If Attribute="TOP-IN":
  - [ ] Change Other\_Indexes to Comm-IN
  - [ ] Change Attribute to "IO"
- [ ] If Attribute="TOP-OUT":
  - [ ] Change Other\_Indexes to Comm-OUT
  - [ ] Change Attribute to "IO"
- [ ] Split into separate tables based on Attribute
  - [ ] In the mapping file, look for ~FI\_T(…,Attribute) or Attribute(…) = ~FI\_T(…).  In some cases, Attribute is the literal TIMES variable while in other cases it is a different name.  For example, PRC\_RESID sometimes has Attribute="PRC\_RESID" and sometimes it has Attribute="STOCK".
- [ ] Note: \* can also be used to indicate a wildcard or an operation in some cells

<br/>

Special handling for transformation insert tables (~TFM\_INS, ~TFM\_UPD):

- Duplicate rows with comma-separated values
- Remove blank rows
- Create a column for every region in ~BookRegions\_Map, if it doesn't already exist
- If the table has an AllRegions column:
  - For every region column, fill in missing values by copying from AllRegions
- For each row, collect all region columns and transpose them to get a single column with multiple rows.  The new column is named "VALUE".
  - [ ] Copy all other columns across the new rows.
  - [ ] The original column names are now Region values for a new Region column.
- [ ] Create a TechName column.   For each row, replicate it across all TechNames that match the filters specified by the Pset columns.  Pset\_PN is a name filter with \* as wildcard.
- [ ] Create a Commodity column.  For each row, replicate it across all commodities that match the filters specified by the Cset columns.
  - [ ] This step can be skipped for attributes that don't involve commodities, but it won't hurt either way
- [ ] Now treat it like a ~FI\_T table

<br/>

Special handling for ~TFM\_COMGRP:

- Follow the steps for ~TFM\_INS above
- Defines COM\_GMAP in syssettings.dd
- COM\_GMAP(REG,COM\_GRP,COM) = ~TFM\_COMGRP(Region,Name,Commodity)

<br/>

Special handling for ~StartYear (in SysSettings.xls):

- [x] This table always consists of a single cell.  It has no column labels.
- [x] Change the column label to VALUE

<br/>

Special handling for ~TimePeriods:

- [x] This table should have a single column.  Rename the column to "D".
- [x] Add a column named "B" whose first row is ~StartYear and each subsequent value is the previous value plus the previous D column.
- [x] Add a column named "E" defined as B + D - 1
- [x] Add a column named "M" defined as round((B+E)/2) = B + round((D-1)/2)
- [x] Add a column named "Year" equal to M

<br/>

These tables only involve renaming columns:

~Currencies

~ActivePDef

~BookRegions\_Map

<br/>

Special handling for ~TimeSlices:

- [x] This table is unusual in that it is not really a table, but a set of lists of varying length, corresponding to hierarchy levels
- [ ] For each level after the first (Season), add a list that contains all ways of choosing one value from each of the previous levels and concatenating them into a single string.  This list should be named ALL\_X, e.g. ALL\_DAYNITE.
  - [ ] For each such list, make a second list called PARENT\_X that removes the last level from the value
- [ ] Add a list named TS\_GROUP that contains "ANNUAL", then the entire first list (Season), then each of the ALL\_X lists
  - [ ] At the same time, create a list of the same size named TSLVL that contains "ANNUAL", then "SEASON" repeated to the length of the first list, then X repeated to the length of ALL\_X
- [ ] Add a list named TS\_MAP that concatenates all of the ALL\_X lists
- [ ] Add a list named PARENT that concatenates all of the PARENT\_X lists
- [ ] Make a table named TS\_MAP according to times\_mapping.txt
- [ ] Make a table named TS\_GROUP according to times\_mapping.txt

<br/>

Special handling for ~COMEMI:

- For each row, collect all columns not in ("Region","Year","CommName") and transpose them to get a single column with multiple rows.  The new column is named "EMCB".  The old column names become a column named "Other\_Indexes".
- Create columns for Region and Year if they don't exist and apply the usual missing-value handling to them
- Make a table named VDA\_EMCB according to times\_mapping.txt

<br/>

TODO:

Special handling for ~UC\_Sets:

<br/>

~DefUnits (seems to be unused)

~ImpSettings (undocumented)

~TFM\_Csets

~TFM\_Psets

~COMAGG

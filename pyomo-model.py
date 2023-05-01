# Equation counts
#     Total        E        G        L        N        X        C        B
#        14        8        4        2        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        26       26        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        61       61        0

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x2 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x3 = Var(within=Reals, bounds=(0, 1147.069), initialize=0)
m.x4 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x5 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x6 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x7 = Var(within=Reals, bounds=(0, 6073.7685), initialize=0)
m.x8 = Var(within=Reals, bounds=(0, 2024.5895), initialize=0)
m.x9 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x10 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x11 = Var(within=Reals, bounds=(0, 1147.069), initialize=0)
m.x12 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x13 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x14 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x15 = Var(within=Reals, bounds=(0, 6073.7685), initialize=0)
m.x16 = Var(within=Reals, bounds=(0, 2024.5895), initialize=0)
m.x17 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x18 = Var(within=Reals, bounds=(0, 8000), initialize=0)
m.x19 = Var(within=Reals, bounds=(0, 16000), initialize=0)
m.x20 = Var(within=Reals, bounds=(0, 32000), initialize=0)
m.x21 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x22 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x23 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x24 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x25 = Var(within=Reals, bounds=(0, None), initialize=0)
m.x26 = Var(within=Reals, bounds=(0, None), initialize=0)

m.obj = Objective(sense=minimize, expr=m.x1)

m.e1 = Constraint(expr=-m.x1 + m.x23 + m.x24 + m.x25 - m.x26 == 0)
m.e2 = Constraint(
    expr=0.5718820861678005 * m.x21 + 0.3718820861678005 * m.x22 - m.x25 == 0
)
m.e3 = Constraint(
    expr=10.2469507659596 * m.x21 + 9.759000729485335 * m.x22 - m.x24 == 0
)
m.e4 = Constraint(
    expr=8.007782294761386 * m.x21 + 8.30291972521687 * m.x22 - m.x26 == 0
)
m.e5 = Constraint(
    expr=-2.75 * m.x3
    + 2.75 * m.x4
    + 8888 * m.x5
    + 2222 * m.x6
    + 2 * m.x7
    + 2.5 * m.x8
    + 3 * m.x9
    - 5.113378684807256 * m.x11
    + 5.113378684807256 * m.x12
    + 16526.439909297053 * m.x13
    + 4131.609977324263 * m.x14
    + 3.7188208616780045 * m.x15
    + 4.648526077097506 * m.x16
    + 5.578231292517007 * m.x17
    - m.x23
    == 0
)
m.e6 = Constraint(expr=m.x2 - 0.95 * m.x21 <= 0)
m.e7 = Constraint(expr=m.x10 - 0.95 * m.x21 - 0.95 * m.x22 <= 0)
m.e8 = Constraint(expr=-m.x2 - m.x3 + m.x4 + m.x6 + m.x7 + m.x8 + m.x9 >= 0)
m.e9 = Constraint(expr=m.x2 + m.x5 >= 13413.96)
m.e10 = Constraint(expr=-m.x10 - m.x11 + m.x12 + m.x14 + m.x15 + m.x16 + m.x17 >= 0)
m.e11 = Constraint(expr=m.x10 + m.x13 >= 13413.96)
m.e12 = Constraint(expr=0.1 * m.x7 + 0.2 * m.x15 - m.x18 == 0)
m.e13 = Constraint(expr=0.1 * m.x8 + 0.2 * m.x16 - m.x19 == 0)
m.e14 = Constraint(expr=0.1 * m.x9 + 0.2 * m.x17 - m.x20 == 0)

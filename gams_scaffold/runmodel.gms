$OFFLISTING
$if NOT set solve_with $SET solve_with CBC
$if NOT set run_name $SET run_name SCENARIO
$hiddenCall gams SCENARIO.RUN parmfile=gams.opt LP=%SOLVE_WITH% --run_name=%RUN_NAME% gdx=%RUN_NAME% O=%RUN_NAME%.lst action=c
$terminate

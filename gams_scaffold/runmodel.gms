$OFFLISTING
$if NOT set solve_with $SET solve_with CBC
$if NOT set run_name $SET run_name scenario
$hiddenCall gams scenario.run parmfile=gams.opt LP=%SOLVE_WITH% --run_name=%RUN_NAME% gdx=%RUN_NAME% O=%RUN_NAME%.lst action=c filecase=4
$terminate

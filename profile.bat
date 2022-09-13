call env/scripts/activate

pip install py-spy

py-spy record -o speedscope.json --format speedscope --subprocesses python times_excel_reader.py

start "" https://www.speedscope.app/

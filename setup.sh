# create virtual environment
virtualenv -p python3 env

# activate virtual environment
source env/bin/activate

# install packages using provided requirements file
pip install -r requirements.txt
python tests/package_test.py

# install jupyter lab plotly extension
jupyter labextension install @jupyterlab/plotly-extension

# deactivate virtual environment
deactivate


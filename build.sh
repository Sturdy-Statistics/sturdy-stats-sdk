rm -rf dist build sturdy_stats_sdk.egg-info
python setup.py bdist_wheel
python setup.py sdist
twine upload dist/*

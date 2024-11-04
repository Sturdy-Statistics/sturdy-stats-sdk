rm -rf dist build sturdy_stats_sdk.egg-info
python setup.py sdist bdist_wheel
twine upload dist/*

"""Model steps that run in a child Python process.

They import only numpy / pandas / pyarrow / scipy and the model library, never pyspark or src.common,
so they also run in a Python 3.9 environment with the kit's pinned versions
(tools/python/python.yaml: statsmodels 0.12.2, xgboost 1.7.3, pandas 1.5.2). run.py selects the
interpreter with --model-python. Each job reads <job>.json and writes <job>.result.json.
"""

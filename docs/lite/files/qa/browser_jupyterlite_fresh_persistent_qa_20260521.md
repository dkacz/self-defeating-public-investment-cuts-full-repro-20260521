# Browser JupyterLite fresh/persistent QA 2026-05-21

## fresh browser context
- url_opened: http://127.0.0.1:8797/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
- body_contains_new_notebook_or_title: True
- body_contains_old_notebook: False
- response_contains_new_notebook: True
- response_contains_old_notebook: False
- interesting_responses: [('http://127.0.0.1:8797/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/jupyter-lite.ipynb', 200), ('http://127.0.0.1:8797/lite/lab/jupyter-lite.ipynb', 200), ('http://127.0.0.1:8797/lite/api/contents/all.json', 200), ('http://127.0.0.1:8797/lite/api/contents/notebooks/all.json', 200), ('http://127.0.0.1:8797/lite/api/contents/notebooks/all.json', 200), ('http://127.0.0.1:8797/lite/files/notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/files/notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/api/contents/code/all.json', 200)]
- storage_keys: {"local":[],"session":[]}
- status: PASS
## persistent browser profile
- url_opened: http://127.0.0.1:8797/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb
- body_contains_new_notebook_or_title: True
- body_contains_old_notebook: False
- response_contains_new_notebook: True
- response_contains_old_notebook: False
- interesting_responses: [('http://127.0.0.1:8797/lite/lab/index.html?path=notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/lab/jupyter-lite.ipynb', 200), ('http://127.0.0.1:8797/lite/jupyter-lite.ipynb', 200), ('http://127.0.0.1:8797/lite/api/contents/all.json', 200), ('http://127.0.0.1:8797/lite/api/contents/notebooks/all.json', 200), ('http://127.0.0.1:8797/lite/api/contents/notebooks/all.json', 200), ('http://127.0.0.1:8797/lite/files/notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/files/notebooks/self_defeating_public_investment_cuts_full_repro_20260521.ipynb', 200), ('http://127.0.0.1:8797/lite/api/contents/code/all.json', 200)]
- storage_keys: {"local":[],"session":[]}
- status: PASS
## static payload
- new_notebook_http_status: 200
- old_notebook_http_status: 404
- status: PASS

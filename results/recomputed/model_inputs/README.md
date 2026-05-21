# Recomputed Runtime Model Inputs

This folder is recreated by the public estimator during each run.

It contains the model-input panel and transformation table used by the downstream
estimation scripts after the source-rebuild step. The source-rebuild QA compares
these runtime files with the immutable validation bundle shipped with the release,
but this directory itself is recomputed output and can be deleted before the next
run.

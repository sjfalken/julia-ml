#!/bin/bash

set -e
set -x

gsutil cp Project.toml Manifest.toml conv.jl gs://sjfalken-learning/

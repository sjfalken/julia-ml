#!/usr/bin/env bash

set -e
set -x

if [ "$#" -ne 0 ]; then
    echo "illegal num parameters"
fi
# cp conv.jl ~/sjfalken-learning
JOBCMD=""
sed -r 's/(^.+JOBCMD\": +\").*(\")/\1'"$JOBCMD"'\2/' batch-config.json > /tmp/job.json
# sed -r 's/(^.+JOBCMD\": +\").*(\")/\1\/app\/julia-1.9.0\/bin\/julia ./conv.jl\2/' batch-config.json > /tmp/job.json

gcloud batch jobs submit --config /tmp/job.json --location us-east4
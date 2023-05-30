#!/usr/bin/env bash

if [ "$#" -ne 0 ]; then
    echo "illegal num parameters"
fi
cp conv.jl ~/sjfalken-learning
sed -r 's/(^.+JOBCMD\": +\").*(\")/\1julia conv.jl\2/' batch-config.json > /tmp/job.json

gcloud batch jobs submit --config /tmp/job.json --location us-east4
#!/usr/bin/env bash

set -ex


python3 -m markov.rover_agent --aws-region=us-east-1 --model-s3-bucket=beep --model-s3-prefix=boop

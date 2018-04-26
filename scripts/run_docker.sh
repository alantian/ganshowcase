#!/usr/bin/env bash

#######################################
# Bash3 Boilerplate Start
# copied from https://kvz.io/blog/2013/11/21/bash-best-practices/

set -o errexit
set -o pipefail
set -o nounset
# set -o xtrace

# Set magic variables for current file & dir
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
__file="${__dir}/$(basename "${BASH_SOURCE[0]}")"
__base="$(basename ${__file} .sh)"
__root="$(cd "$(dirname "${__dir}")" && pwd)" # <-- change this as it depends on your app

arg1="${1:-}"
# Bash3 Boilerplate End
#######################################

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

docker_tag="local:dev-latest"

cmd="$@"
cmd="${cmd:-/bin/bash}"

nvidia-docker run -it --rm \
  -v "${__dir}/../code":/code \
  -v "${__dir}/../scratch":/scratch \
  -v `readlink -f "${__dir}/../scratch2"`:/scratch2 \
  --user="`id -u`:`id -g`" \
  -w /code \
  -e HOME="/tmp" \
  -e CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  "${docker_tag}" \
  "$@"

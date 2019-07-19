#!/bin/bash
set -e

if [ "${1:0:1}" = '-' ]; then
	set -- python /optimizer/optimizer.py "$@"
fi

exec "$@"
#!/usr/bin/env bash
fswatch -o kernels | xargs -n1 -I{} ./scripts/build_metal.sh

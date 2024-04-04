#!/usr/bin/env bash

### Script to bulk clone/fork all of a GitHub organisation's repositories ###

set -o pipefail
# set -xv

### Variables ###

PARALLEL_THREADS="8"

### Checks ###

GITHUB_CLI=$(which gh)
if [ ! -x "$GITHUB_CLI" ]; then
    echo "The GitHub CLI was NOT found in your PATH"; exit 1
fi

PARALLEL_CMD=$(which parallel)
if [ ! -x "$PARALLEL_CMD" ]; then
    echo "The GNU parallel command was NOT found in your PATH"
    echo "On macOS you can install with homebrew using:"
    echo "  brew install parallel"; exit 1
fi

_usage() {
    echo "Script has two modes of operation:"
    echo "Usage: $0 clone [ src github org ]"
    echo "       $0 fork [ src github org ] [ dst github org ]"; exit 1
}

# Source repository specification is entirely optional
# (if unspecified uses your personal profile repos)

# Setup the two different parameters of clone operations
if  { [ $# -eq 1 ] || [ $# -eq 2 ]; } && [ "$1" = "clone" ]; then
    SOURCE_GITHUB_ORG="$2"

# Setup the two different parameters of fork operations
elif  [ $# -eq 2 ] && [ "$1" = "fork" ]; then
    SOURCE_GITHUB_ORG="$2"
    FLAGS="--default-branch-only --clone --remote"
elif  [ $# -eq 3 ] && [ "$1" = "fork" ]; then
    SOURCE_GITHUB_ORG="$2"
    TARGET_GITHUB_ORG="$3"
    FLAGS="--default-branch-only --org $TARGET_GITHUB_ORG --clone --remote"
else
    _usage
fi

OPERATION="$1"
echo "Parallel threads: $PARALLEL_THREADS"

### Functions ###

auth_check() {
    if ! ("$GITHUB_CLI" auth status); then
        echo "You are not logged into GitHub"
        echo "Use the command: gh auth login"; exit 1
    fi
}

### Operations m###

# Make sure we are logged into GitHub
auth_check

# List all the repositories in the source ORG
# Then clone/fork them (to the target ORG if forking)
"$GITHUB_CLI" repo list "$SOURCE_GITHUB_ORG" \
    --limit 4000 --json nameWithOwner --jq '.[].nameWithOwner' | \
    "$PARALLEL_CMD" --j "$PARALLEL_THREADS" "$GITHUB_CLI" repo \
    "$OPERATION" "$FLAGS"

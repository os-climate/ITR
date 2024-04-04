#!/usr/bin/env bash

### Script to bulk refresh a directory containing repositories ###

# shellcheck disable=SC2317

set -o pipefail
# set -xv

### Variables ###

PARALLEL_THREADS="8"

### Checks ###

GIT_CMD=$(which git)
export GIT_CMD
if [ ! -x "$GIT_CMD" ]; then
    echo "GIT was not found in your PATH"; exit 1
fi

if [ $# -ne 0 ]; then
    echo "Usage: $0"; exit 1
fi

echo "Parallel threads: $PARALLEL_THREADS"

### Functions ###

check_is_repo() {
    CURRENT_DIR=$(basename "$PWD")
    # Check current directory is a GIT repository
    "$GIT_CMD" status > /dev/null 2>&1
    if [ $? -eq 128 ]; then
        echo "Skipping folder NOT a git repository: $CURRENT_DIR"
        return 1
    else
        printf "Processing: %s -> " "$CURRENT_DIR"
        # Figure out which of the two options is the primary branch name
        GIT_MAIN=$("$GIT_CMD" branch -l main \
            master --format '%(refname:short)')
        export GIT_MAIN
        return 0
    fi
}

count_repos() {
    # Count the number of GIT repositories
    REPOS="0"
    FOLDERS=$(find . -type d -depth 1)
    for FOLDER in $FOLDERS; do
        TARGET=$(basename "$FOLDER")
        if (check_is_repo "$TARGET"); then
            REPOS=$((REPOS+1))
        fi
    done
    FOUND=$(echo "$FOLDERS" | wc -w)
    echo "Found: $FOUND directories, $REPOS git repositories"
}

check_if_fork() {
    # Checks for both upstream and origin
    UPSTREAM_COUNT=$(git remote | \
        grep -E -e 'upstream|origin' -c)
    if [ "$UPSTREAM_COUNT" -eq 2 ]; then
        # Repository is a fork
        return 0
    else
        return 1
    fi
}

checkout_head_branch() {
    CURRENT_BRANCH=$("$GIT_CMD" branch --show-current)
    HEAD_BRANCH=$("$GIT_CMD" rev-parse --abbrev-ref HEAD)
    # Only checkout HEAD if not already on that branch
    if [ "$CURRENT_BRANCH" != "$HEAD_BRANCH" ]; then
        # Need to swap branch in this repository
        if ("$GIT_CMD" checkout "$HEAD_BRANCH" > /dev/null 2>&1); then
            printf "switched to %s -> " "$HEAD_BRANCH"
            return 0
        else
            echo "Error checking out $HEAD_BRANCH"
            return 1
        fi
    else
        # Already on the appropriate branch
        printf "%s -> " "$HEAD_BRANCH"
        return 0
    fi
}

update_repo() {
    if ! (check_if_fork); then
        printf "updating clone -> "
        if ("$GIT_CMD" pull > /dev/null 2>&1;); then
            echo "Done."
            return 0
        else
            echo "Error."
            return 1
        fi
    else
        # Repository is a fork
        printf "resetting fork -> "
        if ("$GIT_CMD" fetch upstream > /dev/null 2>&1; \
            "$GIT_CMD" reset --hard upstream/"$GIT_MAIN" > /dev/null 2>&1; \
            "$GIT_CMD" push origin "$GIT_MAIN" --force > /dev/null 2>&1); then
            echo "Done."
            return 0
        else
            echo "Error."
            return 1
        fi
    fi
}

change_dir_error() {
    echo "Could not change directory"; exit 1
}

refresh_repo() {
    # Change into the target directory
    cd "$1" || change_dir_error
    # Check current directory is a GIT repository
    if check_is_repo; then
        if (checkout_head_branch); then
            # Update the repository
            update_repo
        fi
    fi
    # Change back to parent directory
    cd .. || change_dir_error
}

# Make functions available to GNU parallel
export -f refresh_repo check_if_fork update_repo \
    checkout_head_branch check_is_repo change_dir_error

### Operations ###

CURRENT_DIR=$(basename "$PWD")
echo "Processing all GIT repositories in: $CURRENT_DIR"
find -- * -depth 0 -type d | parallel -j "$PARALLEL_THREADS" --env _ refresh_repo :::
echo "Script completed"; exit 0

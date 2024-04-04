#!/usr/bin/env bash

### Script to bulk raise a PR in multiple repositories ###

# shellcheck disable=SC2317

set -o pipefail
# set -xv

### Variables ###

CONDITIONAL="yes"
PARALLEL_THREADS="8"

### Checks ###

GIT_CMD=$(which git)
if [ ! -x "$GIT_CMD" ]; then
    echo "GIT was not found in your PATH"; exit 1
fi
export GIT_CMD

GITHUB_CLI=$(which gh)
if [ ! -x "$GITHUB_CLI" ]; then
    echo "The GitHub CLI was not found in your PATH"; exit 1
fi
export GITHUB_CLI

echo "Parallel threads: $PARALLEL_THREADS"

### Functions ###

auth_check() {
    if ! (gh auth status); then
        echo "You are not logged into GitHub"
        echo "Use the command: gh auth login"; exit 1
    fi
}

change_dir_error() {
    echo "Could not change directory"; exit 1
}

check_if_main() {
    # Figure out which of the two options is the primary branch name
    PRIMARY_BRANCH=$("$GIT_CMD" branch -l main \
        master --format '%(refname:short)')
    export PRIMARY_BRANCH
    if [ "$PRIMARY_BRANCH" = "main" ]; then
        return 0
    else
        return 1
    fi
}

check_is_repo() {
    DIRECTORY="$1"
    cd "$DIRECTORY" || change_dir_error
    # Check current directory is a GIT repository
    "$GIT_CMD" status > /dev/null 2>&1
    RETURN_CODE="$?"
    cd .. || change_dir_error
    if [ "$RETURN_CODE" -eq 128 ]; then
        echo "Folder is NOT a git repository: $DIRECTORY"
        return 1
    else
        return 0
    fi
}

perform_repo_actions() {

    # Define variables
    REPO="$1"
    SLEEP_TIME="60"

    # Only take action if pre-condition is met
    if [ ! -f "$REPO"/.pre-commit-config.yaml ]; then

        # Actions/changes to perform automatically in repository
        echo "No pre-commit config: $REPO"
        cp .pre-commit-config.yaml "$REPO"

        ### Values for GIT operations
        BRANCH="implement-minimal-precommit"
        TITLE="Implement minimal pre-commit configuration for pre-commit.ci"
        BODY="This will satisfy pre-commit.ci and prevent merges from blocking"

        ### Raise a PR with upstream/main
        cd "$REPO" || change_dir_error
        "$GIT_CMD" pull
        "$GIT_CMD" checkout -b "$BRANCH"
        "$GIT_CMD" add .pre-commit-config.yaml
        "$GIT_CMD" commit -as -S -m "Chore: $TITLE" --no-verify
        "$GIT_CMD" push
        PR_URL=$("$GITHUB_CLI" pr create --title "$TITLE" --body "$BODY")
        PR_NUMBER=$(basename "$PR_URL")
        echo "Pull request #$PR_NUMBER URL: $PR_URL"
        echo "Sleeping..."
        sleep "$SLEEP_TIME"
        "$GITHUB_CLI" pr merge "$URL" --delete-branch --merge
        "$GIT_CMD" push origin --delete "$BRANCH" > /dev/null 2>&1 &
        "$GIT_CMD" push upstream --delete "$BRANCH" > /dev/null 2>&1 &
        # Change back to parent directory
        cd .. || change_dir_error

        ### Optionally remove branches (if not handled automatically by repo settings)
        # if (git ls-remote --heads origin refs/heads/"$BRANCH"); then
        #      echo "Attempting deletion of branch: origin/$BRANCH"
        #      git push origin --delete "$BRANCH"
        # fi
        # if (git ls-remote --heads upstream refs/heads/"$BRANCH"); then
        #   echo "Attempting deletion of branch: upstream/$BRANCH"
        #      git push upstream --delete "$BRANCH"
        # fi
    fi
}

# Export functions for use by GNU parallel tool
export -f perform_repo_actions

### Operations ###

auth_check

# Only used if repository operations are optional
if [ "$CONDITIONAL" = "yes" ]; then

    # Count the number of GIT repositories
    REPOS="0"

    ### Modify the code below with conditions ###

    FOLDERS=$(find . -type d -depth 1)
    for FOLDER in $FOLDERS; do
        TARGET=$(basename "$FOLDER")
        if (check_is_repo "$TARGET"); then
            REPOS=$((REPOS+1))
            if [ ! -f "$TARGET"/.pre-commit-config.yaml ]; then
                COUNTER=$((COUNTER +1 ))
                TARGETS+=" $TARGET"
                echo "FOUND!"
            fi
        fi
    done
    UPDATES=$(echo "$TARGETS" | wc -w)
    PROCESSED=$(echo "$FOLDERS" | wc -w)
    echo "$PROCESSED directories, $REPOS repositories"
fi

if ! [ "$UPDATES" -eq "0" ]; then
    echo "$UPDATES repositories require updates"
    # Invoke GNU parallel to update the repository

    echo "Script completed"; exit 0
else
    echo "No repository operations were required"; exit 1
fi

# Check if repository is archived or read-only
#find -depth 0 -type d -print0 | while read -r -d $'\0' REPO; do
#    # Should migrate this to a bunch of parallel operations
#    parallel -j "$PARALLEL_THREADS" perform_repo_actions ::: "$REPO"
#    # parallel -j "$PARALLEL_THREADS" --env _ perform_repo_actions ::: "$REPO"
#done

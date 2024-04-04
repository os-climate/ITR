#!/usr/bin/env bash

### Script to bulk clone/fork a GitHub organisation's repositories ###

set -o pipefail
# set -xv

### Variables ###

PARALLEL_THREADS="8"
ORG_REPO_LIMIT="1000"

### Checks ###

GITHUB_CLI=$(which gh)
if [ ! -x "$GITHUB_CLI" ]; then
    echo "The GitHub CLI was not found in your PATH"; exit 1
fi

if [ $# -ne 1 ]; then
    echo "Usage: $0 [ dst github org ]"; exit 1
else
    TARGET_GITHUB_ORG="$1"
fi

echo "Parallel threads: $PARALLEL_THREADS"

### Functions ###

auth_check() {
    if ! ("$GITHUB_CLI" auth status); then
        echo "You are not logged into GitHub"
        echo "Use the command: gh auth login"; exit 1
    fi

}

empty_the_org() {
    ORG_REPO_LIST=$("$GITHUB_CLI" repo list "$TARGET_GITHUB_ORG" \
    --limit "$ORG_REPO_LIMIT" | awk '{print $1}' \
    | grep "$TARGET_GITHUB_ORG")
    REPO_COUNT=$(echo "$ORG_REPO_LIST" | wc -w)
    if [ "$REPO_COUNT" -gt 0 ]; then
        echo "Repositories to remove: $REPO_COUNT"
        for REPO in $ORG_REPO_LIST; do
            # Attempt an initial deletion, which may fail
            if ! ("$GITHUB_CLI" repo delete --yes "$REPO"); then
                # We need to enable the GitHub CLI tool to delete repositories
                "$GITHUB_CLI" auth refresh -h github.com -s delete_repo
                # Token provided, enter in browser, then try operation again
                if ! ("$GITHUB_CLI" repo delete --yes "$REPO"); then
                    echo "Something went wrong; check account/permissions"
                    exit 1
                fi
            fi
        done
    else
        echo "There are no repositories in that ORG to delete"; exit 1
    fi
}

### Operations m###

auth_check

for RESULT in $("$GITHUB_CLI" org list | \
    grep -v 'Showing * of * organizations'); do
    # Try and match/verify that the organisation exists
    if [ "$RESULT" = "$TARGET_GITHUB_ORG" ]; then
        empty_the_org
        echo "Script completed successfully"; exit 0
    fi
done
echo "You do not appear to be a member of GitHub ORG: $TARGET_GITHUB_ORG"
exit 1

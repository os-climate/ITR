import git
from typing import List

def get_unsigned_commits(repo: git.Repo) -> List[git.Commit]:
    unsigned_commits = []
    for commit in repo.iter_commits():
        if not commit.message.__contains__("Signed-off-by"):
            unsigned_commits.append(commit)
    return unsigned_commits

def get_authors(unsigned_commits: List[git.Commit]) -> List[str]:
    authors = [str(commit.author) for commit in unsigned_commits]
    unique_authors = list(set(authors))
    return unique_authors

def get_authors_unsigned_commits(repo) -> List[str]:
    unsigned_commits = get_unsigned_commits(repo)
    return get_authors(unsigned_commits)

def add_dco_signoff_to_file(repo: git.Repo, name: str, authors: List[str], email_addresses: List[str], declaration: str = ""):
    """
    repo: The repository for which to add DCO sign-offs. Make sure to have checked out the correct branch.
    name: Name used to identify the one declaring sign-off statement.
    gh_names: List of names under which the unsigned commits were committed - can be multiple for the same user.
    email_addresses: List of any past email address used by user in relation to the unsigned commits. May be an empty list.
    """
    if not declaration:
        declaration = f"I, {name}, hereby sign-off-by all of my past commits to this repo subject to the Developer " + \
                      f"Certificate of Origin (DCO), Version 1.1. "
        if email_addresses:
            declaration += f"In the past I have used emails: {email_addresses}. "
    unsigned_commits = get_unsigned_commits(repo)
    authors_commits = [commit for commit in unsigned_commits if str(commit.author) in authors]
    with open(name + '.txt', 'w') as file:
        file.write(declaration + "\n")
        for commit in authors_commits:
            message = commit.message.replace('\n', ' ')
            file.write(f"{commit.hexsha} {message}\n")
#!/usr/bin/env python3
"""Fetch and display GitHub repository statistics for all repos in the yazhi-lem org."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from typing import Any

ORG = "yazhi-lem"
API_BASE = "https://api.github.com"


def _get(url: str, token: str | None) -> Any:
    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        print(f"GitHub API error {exc.code}: {exc.reason}", file=sys.stderr)
        sys.exit(1)


def fetch_repos(token: str | None) -> list[dict]:
    repos: list[dict] = []
    page = 1
    while True:
        url = f"{API_BASE}/orgs/{ORG}/repos?per_page=100&page={page}"
        batch = _get(url, token)
        if not batch:
            break
        repos.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return repos


def fetch_languages(repo_full_name: str, token: str | None) -> dict[str, int]:
    url = f"{API_BASE}/repos/{repo_full_name}/languages"
    return _get(url, token)



def build_row(repo: dict, languages: dict[str, int]) -> dict:
    top_lang = max(languages, key=lambda k: languages[k]) if languages else repo.get("language") or "—"
    return {
        "name": repo["name"],
        "description": repo.get("description") or "",
        "language": top_lang,
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "watchers": repo.get("watchers_count", 0),
        "size_kb": repo.get("size", 0),
        "created_at": (repo.get("created_at") or "")[:10],
        "updated_at": (repo.get("updated_at") or "")[:10],
        "url": repo.get("html_url", ""),
        "languages": languages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Show GitHub repository statistics for the {ORG} organisation"
    )
    parser.add_argument(
        "--token",
        metavar="GITHUB_TOKEN",
        help="Personal access token for authenticated requests (higher rate limit)",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output stats as JSON instead of a human-readable table",
    )
    parser.add_argument(
        "--no-languages",
        action="store_true",
        help="Skip the per-repo language breakdown API call",
    )
    args = parser.parse_args()

    print(f"Fetching repos for organisation: {ORG} …", file=sys.stderr)
    repos = fetch_repos(args.token)
    print(f"Found {len(repos)} repo(s).", file=sys.stderr)

    rows = []
    for repo in repos:
        languages: dict[str, int] = {}
        if not args.no_languages:
            languages = fetch_languages(repo["full_name"], args.token)
        rows.append(build_row(repo, languages))

    if args.as_json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print(f"  {ORG} — repository statistics")
        print(f"{'='*60}\n")
        for row in rows:
            print(f"  {row['name']}")
            print(f"    Description : {row['description'] or '(none)'}")
            print(f"    URL         : {row['url']}")
            print(f"    Language    : {row['language']}")
            if row["languages"]:
                lang_str = ", ".join(
                    f"{k} ({v:,} B)" for k, v in sorted(row["languages"].items(), key=lambda x: -x[1])
                )
                print(f"    Languages   : {lang_str}")
            print(f"    Stars       : {row['stars']}")
            print(f"    Forks       : {row['forks']}")
            print(f"    Open issues : {row['open_issues']}")
            print(f"    Watchers    : {row['watchers']}")
            print(f"    Size        : {row['size_kb']:,} KB")
            print(f"    Created     : {row['created_at']}")
            print(f"    Updated     : {row['updated_at']}")
            print()
        print(f"Total repos: {len(rows)}")


if __name__ == "__main__":
    main()

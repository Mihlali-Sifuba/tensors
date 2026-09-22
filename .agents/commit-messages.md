# Commit messages, for implementation agents

## Follow the repository's Git conventions

[`docs/git-conventions.md`](../docs/git-conventions.md) is the authoritative
source for Git and commit-message conventions in this repository. Read it
before committing and follow it. This file adds one policy; it does not
restate, replace or override that document.

In particular, that document requires:

- **Atomic commits.** One logical change per commit. Do not combine unrelated
  features, fixes, refactors, documentation updates or maintenance work.
- **The header format** `<type>(<scope>): <summary>`, with a type from its
  list and a scope naming the affected subsystem rather than a filename.
- **A blank line** between the header and the body.
- **UTF-8 `•` bullet characters** for the body's points.
- **Meaningful, descriptive messages** that summarise the substantive changes
  rather than the implementation noise.
- **Its branch-history and merge conventions**, including keeping a branch's
  commits atomic and integrating completed branches with `--no-ff`.

## Attribution

**AI agents must not add themselves as commit co-authors.**

- Do not add a `Co-authored-by` trailer crediting an AI agent.
- Do not append an agent name, a model name or an AI service name as a
  co-author.
- Do not append automatic agent-credit trailers or attribution statements of
  any kind to a commit message.

This applies to the current work and to every subsequent agent-authored
commit. A commit's author is the human the work is for.

Do not rewrite historical commits solely to remove attribution that is
already there.

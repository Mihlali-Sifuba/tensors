# Git conventions

These conventions keep the repository history readable and make each change easy
to understand, review, and revert.

## Commits

Each commit must represent one logical change. Do not combine unrelated features,
fixes, refactors, documentation updates, or maintenance work in the same commit.

Use this commit-message structure:

```text
<type>(<scope>): <header>

• <change introduced>
• <change introduced>
• <change introduced>
```

For example:

```text
feat(tensor): Add broadcasting support

• Add broadcast-shape resolution for compatible dimensions
• Add singleton-dimension expansion
• Raise an error for incompatible tensor shapes
• Add tests for multidimensional broadcasting
```

### Header

The header must:

- Follow `<type>(<scope>): <summary>`.
- Be concise and descriptive.
- Describe the primary change introduced by the commit.
- Prefer imperative wording.

Use these commit types where appropriate:

- `feat` — new functionality
- `fix` — bug fixes
- `refactor` — internal restructuring without intended behaviour changes
- `docs` — documentation changes
- `test` — test additions or modifications
- `perf` — performance improvements
- `chore` — tooling, configuration, dependencies, or maintenance

The scope should identify the affected subsystem, component, or area rather than
simply repeating a filename.

### Body

The commit body must:

- Be separated from the header by a blank line.
- Use UTF-8 bullet characters: `•`.
- Summarize the substantive changes introduced by the commit.
- Avoid unnecessary implementation noise.

## Branches

Each branch should represent one focused piece of work. Use descriptive lowercase
names with hyphen-separated words and a type prefix.

For example:

```text
feature/tensor-broadcasting
fix/incorrect-flat-index
refactor/tensor-storage
docs/git-conventions
test/broadcasting-cases
experiment/native-vector-add
chore/update-tooling
```

Suggested branch prefixes:

- `feature/`
- `fix/`
- `refactor/`
- `docs/`
- `test/`
- `experiment/`
- `chore/`

A branch name should communicate its purpose without requiring someone to inspect
its commits.

## Branch history

A branch may contain multiple commits, but every commit should remain atomic and
meaningful.

Before merging a completed branch, clean up unnecessary history where practical.
Temporary `WIP`, debugging, typo-only, or accidental commits should not remain in
the final branch history when they can reasonably be consolidated or amended.

Do not squash an entire branch merely to produce one commit when it already
contains a meaningful sequence of atomic commits.

When appropriate, keep a private or individually owned working branch current by
rebasing it onto the latest target branch. Do not casually rebase shared branches,
because rebasing rewrites commit history.

## Merging

Do not use fast-forward merges for completed branches. Integrate them using an
explicit merge commit so the branch boundary remains visible in Git history.

Prefer:

```bash
git merge --no-ff <branch>
```

over a fast-forward merge. This preserves both:

- The individual commits that make up the branch.
- A distinct merge commit showing when the logical unit of work was integrated.

Merge commits should contain a meaningful summary of what the branch introduced.
For example:

```text
Merge feature/tensor-broadcasting

• Add tensor broadcasting support
• Add singleton-dimension expansion
• Add broadcasting validation
• Add associated tests
```

The resulting history should make it easy to identify both individual
implementation changes and the larger branch-level units of work they belong to.

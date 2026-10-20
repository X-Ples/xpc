# X+ Liblary Developer Rules & Guide

This document is **required reading** for anyone who wants to contribute a
library to **[X-Ples/Liblary](https://github.com/X-Ples/Liblary)** — the
official X+ registry. It covers **how to create a library** and **the rules you
must follow**.

> [!IMPORTANT]
> Opening a Pull Request to this repository means you **agree** to every rule in
> this document. Maintainers may reject any PR that violates them without
> further discussion.

---

## Table of Contents

1. [How to Create a Library](#1-how-to-create-a-library)
2. [`registry.toml` Format](#2-registrytoml-format)
3. [Mandatory Rules](#3-mandatory-rules)
4. [Strictly Prohibited](#4-strictly-prohibited)
5. [Quality Standards](#5-quality-standards)
6. [Pre-PR Checklist](#6-pre-pr-checklist)
7. [Review Process](#7-review-process)
8. [Sanctions](#8-sanctions)

---

## 1. How to Create a Library

> [!NOTE]
> The flow is always the same: **Fork → create your library folder → register it
> in `registry.toml` → test → Pull Request.** There are no shortcuts.

### Step 1 — Fork the repository (REQUIRED)

Open <https://github.com/X-Ples/Liblary> and click **Fork**. You may not push
directly to the main repository — every contribution **must** go through a fork
and a Pull Request.

### Step 2 — Clone your fork and create a branch

```bash
git clone https://github.com/<username>/Liblary.git
cd Liblary
git checkout -b lib/<library-name>      # convention: lib/<library-name>
```

### Step 3 — Create your own library folder

One library = one folder, named exactly like the library:

```
<library-name>/
├── <library-name>.xp    # the library source (required)
├── example.xp           # a runnable example program (required)
└── README.md            # description + usage (required)
```

Your library source file must begin with a header comment containing the
library name, its version, and the line `Implicit namespace: <library-name>::`.

### Step 4 — Add / edit `registry.toml` at the root (REQUIRED)

Open `registry.toml` at the **root** of the repository and add an entry for your
library. If the file does not exist yet, create it. Full details are in
[section 2](#2-registrytoml-format).

```toml
[packages.geomplus]
version = "0.1.0"
author = "name (<email>)"
path = "geomplus/geomplus.xp"
size_kb = 0.55
date = "19/Oct/2026"
description = "vec2 geometry operations: distance, midpoint, lerp"
```

> [!WARNING]
> A library **without a `registry.toml` entry will not be accepted**. Only
> **add** your own new entry — never delete or modify entries that belong to
> other contributors.

### Step 5 — Test it like a real user

```bash
xpc check <library-name>/<library-name>.xp     # validate syntax + types
xpc lint  <library-name>/<library-name>.xp     # extra static analysis
xpc run   <library-name>/example.xp            # the example MUST run with correct output
```

Simulate a user who only has the cache:

```bash
cp <library-name>/<library-name>.xp ~/.xpc/libs/
cd /tmp && mkdir test && cd test
# create a main.xp that imports your library with  using = <<<library-name>.xp>>
# and calls its functions, then:
xpc run main.xp
```

> [!TIP]
> Also run `xpc mem-check` and `xpc bounds-check` on your library. PRs that come
> back clean from both audits get approved much faster.

### Step 6 — Commit and push

```bash
git add <library-name>/ registry.toml
git commit -m "add <library-name> v<version> — <short description>"
git push origin lib/<library-name>
```

### Step 7 — Open a Pull Request

On GitHub: your fork → **Compare & pull request** → target `X-Ples/Liblary`,
branch `main`. Fill in the PR description using this template:

```markdown
## New library: <name> v<version>

**What:** <one-sentence summary>
**Why it is useful:** <benefit for X+ users>

**Tested:**
- [ ] xpc check passes for all files
- [ ] example.xp runs and the output matches the documentation
- [ ] registry.toml entry follows the [packages.<name>] format
- [ ] does not modify other contributors' libraries
- [ ] this library is legal and harmless
```

---

## 2. `registry.toml` Format

Every library has exactly **one** `[packages.<name>]` block:

```toml
[packages.geomplus]
version = "0.1.0"
author = "name (<email>)"
path = "geomplus/geomplus.xp"
size_kb = 0.55
date = "19/Oct/2026"
description = "vec2 geometry operations: distance, midpoint, lerp"
```

| Field | Required | Description |
|-------|:--------:|-------------|
| `[packages.<name>]` | ✅ | `lowercase`, unique; must match the folder name and namespace |
| `version` | ✅ | `MAJOR.MINOR.PATCH` format (e.g. `0.1.0`) |
| `author` | ✅ | `name (<email>)` — the email **must be active** (used for warnings & contact) |
| `path` | ✅ | Path to the main file from the repo root, e.g. `<name>/<name>.xp` |
| `size_kb` | ✅ | Size of the main file in KB — **must match reality** |
| `date` | ✅ | Release/update date, `dd/Mon/yyyy` format |
| `description` | ✅ | One clear sentence describing what the library does |

> [!TIP]
> Quick way to compute `size_kb`:
>
> ```bash
> wc -c <name>/<name>.xp | awk '{printf "%.2f\n", $1/1024}'
> ```

> [!IMPORTANT]
> The `author` email **must be valid and active**. Violation warnings are sent to
> this address.

---

## 3. Mandatory Rules

1. **Fork + Pull Request is mandatory.** No direct pushes. Every library must be
   reviewed and **approved (agreed) by a maintainer** before it enters the
   registry.
2. **You must add/edit `registry.toml` at the root** using the correct format.
3. **One folder per library**, and the `path` in `registry.toml` must point to a
   file that actually exists.
4. **You must include a runnable `example.xp`.** PRs whose example fails to run
   will be sent back for revision.
5. **All `.xp` files in your library folder must pass `xpc check`.**
6. **Library names must be unique** and `lowercase`. You may not imitate or
   resemble another library's name to mislead users (*typosquatting*).
7. **A header comment is required**: library name, version, and the line
   `Implicit namespace: <name>::`.
8. **`pub fn` only for the public API.** Internal helper functions have no `pub`.
9. **Updates = a new PR** with a bumped `version` (and an updated `date`). Never
   overwrite an old version without bumping it.
10. **One PR = one library** (or one update). Focused PRs are reviewed faster.
11. **Be honest.** The description, version, size, and author in `registry.toml`
    must be accurate.

---

## 4. Strictly Prohibited

> [!CAUTION]
> Violations of this section are **rejected immediately** and handled according
> to [section 8](#8-sanctions).

### 4.1 Illegal & harmful content

Libraries whose main purpose, or real capability, is any of the following are
prohibited:

- tools for **hacking websites/systems without authorization**, exploitation, or
  account break-ins;
- **DDoS** tools, flooders, botnets, or similar network-attack tools;
- **malware**: viruses, worms, ransomware, spyware, keyloggers, stealers, or
  backdoors;
- **phishing** tools, scams, identity forgery, or data theft;
- **hidden crypto miners** (*cryptojackers*) or code that hijacks users' device
  resources;
- anything that violates the law or copyright, or that can be used to harm
  others.

> [!NOTE]
> **Defensive and educational** security libraries (e.g. hashing, input
> validation, auditing) are still allowed — as long as their purpose is clear,
> they cannot be directly misused to attack, and they are openly documented.

### 4.2 Misusing other people's libraries

- You may **not** modify, overwrite, or delete libraries/entries that belong to
  other contributors.
- You may **not** claim someone else's work as your own (plagiarism). If you
  adapt someone else's code, credit them and respect their license.
- You may **not** slip harmful code in through updates to an existing library.

> [!IMPORTANT]
> **Your job is only to create Liblary — not to damage other people's work.**
> Finished and stable libraries belonging to other contributors **must not be
> touched**. Want to fix or extend one? Open an *issue* or contact its author;
> changes happen only through PRs approved by the owner/maintainer.

### 4.3 Hidden practices

- **No** obfuscated code, or code deliberately made hard to read in order to hide
  its behavior.
- **No** sending user data out (telemetry, network, files) **without clear
  documentation** in the library's `README.md`.
- **No** credentials, tokens, API keys, passwords, or personal data inside the
  repository.
- **No** binary files or large irrelevant files.

---

## 5. Quality Standards

> [!TIP]
> A tidy, small, well-documented library is usually approved in a single review
> round.

- **Follow the stdlib style** (`libs/` in the X+ project): header comment,
  `pub fn` for the API, and `guard` to validate preconditions.
- **Use X+ type safety**: use `Result<T,E>` for operations that can fail and `T?`
  for values that can be empty. Avoid preventable panics.
- **Do not depend on other libraries** without documenting it in `README.md`.
- **Document every public function**: purpose, parameters, and return value.
- **The library's `README.md`** must contain at least: a description, an
  import example (`using = <<name.xp>>`), a list of functions, and the license.
- **Examples must be deterministic** — output should be easy to compare with the
  documentation.
- **Keep it small**; a library should focus on a single purpose.

---

## 6. Pre-PR Checklist

- [ ] I forked the repository and am working on a `lib/<name>` branch
- [ ] My library folder contains `<name>.xp`, `example.xp`, and `README.md`
- [ ] The `[packages.<name>]` entry exists in `registry.toml` at the **root**
- [ ] All registry fields are filled in and accurate (`size_kb`, `date`, active email)
- [ ] `xpc check` passes for all files
- [ ] `example.xp` runs and its output is correct
- [ ] Tested through the cache (`~/.xpc/libs/`) like a normal user
- [ ] No changes to other people's libraries/entries
- [ ] No illegal, harmful, or hidden content
- [ ] No credentials or personal data in any commit

---

## 7. Review Process

1. You open a PR from your fork to `X-Ples/Liblary`, branch `main`.
2. A maintainer checks: registry format, code, example, and rule compliance.
3. If there are problems, the maintainer requests changes — update the same
   branch (the PR updates automatically).
4. After it is **approved (agreed)** and merged, the library is available to all
   users:
   - `xpc search <name>` and `xpc info <name>` find it,
   - `xpc install <name>` pulls it from the registry,
   - `using = <<name.xp>>` downloads it automatically (resolution level 4).

> [!NOTE]
> Only maintainers merge. A PR that stays inactive for a long time after changes
> were requested may be closed.

---

## 8. Sanctions

We act **firmly** on violations:

| Violation | Action |
|-----------|--------|
| Wrong registry format / failing example / quality rules | PR sent back for revision |
| Illegal or harmful content (section 4.1) | **Warning email** to the address in `author` + **PR rejected** |
| Misusing or damaging another person's library (section 4.2) | **Warning email** + **PR rejected** |
| Repeated, deliberate, or severe violations | Library removed from the registry and the contributor blocked from the repository |

> [!WARNING]
> Even an already-merged library **can be removed at any time** if it is later
> found to violate these rules or to endanger users.

---

## Quick Summary

- ✅ **Fork → your own library folder → `registry.toml` → test → PR**
- ✅ Wait for maintainer approval
- ❌ No hacking tools, DDoS, malware, or anything else illegal
- ❌ Never touch or damage other people's libraries
- 📬 Violation = warning email + PR rejected

Thank you for helping build a safe and trustworthy X+ ecosystem.

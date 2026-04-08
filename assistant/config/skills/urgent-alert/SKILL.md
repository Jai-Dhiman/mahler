---
name: urgent-alert
description: Post an urgent email alert to Discord. Called when email triage finds an URGENT email.
version: 0.1.0
author: mahler
license: MIT
metadata:
  hermes:
    tags: [email, discord, alerts, notifications]
    related_skills: [email-triage]
---

## When to use

Use this skill when the email triage process classifies an incoming email as URGENT. It posts a formatted Discord embed to the triage webhook so the user is immediately notified with the key details.

## Procedure

Run the alert script with the full email metadata:

```
python3 ~/.hermes/skills/urgent-alert/scripts/alert.py \
  --from "<from_addr>" \
  --subject "<subject>" \
  --summary "<one sentence summary>" \
  --source "<gmail|outlook>"
```

The script reads `DISCORD_TRIAGE_WEBHOOK` from the environment. If the variable is not set, the script raises a `RuntimeError` and exits non-zero — do not silently skip the alert.

## Arguments

| Argument    | Description                                   |
|-------------|-----------------------------------------------|
| `--from`    | Sender email address                          |
| `--subject` | Email subject line                            |
| `--summary` | One-sentence plain-text summary of the email  |
| `--source`  | Mail source: `gmail` or `outlook`             |

## Success

The script prints `Alert sent.` and exits 0 on a successful POST (HTTP 200 or 204).

## Failure

Any non-200/204 response raises `RuntimeError` with the HTTP status and response body. Missing `DISCORD_TRIAGE_WEBHOOK` also raises `RuntimeError`. Both cases result in a non-zero exit.

# Submissions

Create `submissions/YOUR_TEAM_NAME/` with:

```
submission.enc
meta.json
```

Encrypt your prediction JSON:

```bash
python -m encryption.encrypt predictions.json submissions/YOUR_TEAM_NAME/submission.enc
```

`meta.json` example:

```json
{"team": "YOUR_TEAM_NAME", "model": "My detector", "type": "human", "notes": "Short description"}
```

Open a PR that changes only these two files.

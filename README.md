# Ollama

Local Ollama gateway for Settings → LLMs. Point at your Ollama server URL (default http://localhost:11434). Install from Store → Gateways, then enable to show the Ollama row under Providers & Keys.

Desktop plugin for [UEFN-Ducky](https://github.com/UEFN-Ducky/UEFN-Ducky) (`ollama`).
Install or update from **Settings → Store** in the app — do not install from a zip by hand.

## Build

```bash
py scripts/build_zip.py
```

Writes `deploy/ollama-1.0.11.ducky-plugin.zip` (scripts/ and deploy/ are not packed).

## Secrets

Never commit tokens or keys. The app stores `ollama` locally (DPAPI), not in this package.

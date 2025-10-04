# btc-futures-analysis

## Configuration

`analyze.py` can optionally route Binance API traffic through a Cloudflare Worker.
Provide the worker base URL via the `CF_WORKER_BASE` environment variable to
prepend it to the list of REST hosts without editing the source code.

```bash
export CF_WORKER_BASE="https://<your-subdomain>.workers.dev"
python -u analyze.py
```

If the variable is unset or empty the script will continue to use the default
Binance endpoints.

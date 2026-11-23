from pathlib import Path
import secrets
root = Path(__file__).resolve().parent
path = root / ".env"
if path.exists():
    print(".env already exists; not overwriting")
else:
    path.write_text("DENGU_API_TOKEN=" + secrets.token_urlsafe(40) + "\n", encoding="utf-8")
    print("Created .env with a random local API token. Keep it private.")

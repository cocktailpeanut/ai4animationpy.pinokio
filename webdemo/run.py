import os

import uvicorn

from server import app


HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))


def main():
    print(f"http://{HOST}:{PORT}", flush=True)
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=os.environ.get("UVICORN_LOG_LEVEL", "warning"),
    )


if __name__ == "__main__":
    main()

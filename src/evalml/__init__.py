from dotenv import load_dotenv

load_dotenv()

from .cli import cli  # noqa: E402

__all__ = ["cli"]

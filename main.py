from dotenv import load_dotenv

from training.utils.logger import get_logger


def main():
    load_dotenv(override=False)
    print("Hello from training-module!")


if __name__ == "__main__":
    logger = get_logger("training.entrypoint.main")
    try:
        main()
    except Exception:
        logger.exception("Unhandled exception in main entrypoint")
        raise

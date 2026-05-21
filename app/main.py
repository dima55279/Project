import argparse

from app.indexing.indexer import Indexer
from app.modes.chat_mode import ChatMode
from app.modes.csv_mode import CSVMode


parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    required=True,
    choices=["index", "chat", "csv"]
)

parser.add_argument("--question")
parser.add_argument("--input")

args = parser.parse_args()


if args.mode == "index":

    indexer = Indexer()
    indexer.build(clear_db=True)


elif args.mode == "chat":

    chat = ChatMode()

    result = chat.ask(args.question)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nDOCUMENTS:\n")
    print(result["documents"])


elif args.mode == "csv":

    csv_mode = CSVMode()

    csv_mode.run(args.input)

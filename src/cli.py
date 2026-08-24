import json

from src.mapper import default_mapper


def main() -> None:
    mapper = default_mapper()
    print("请输入一条消息，输入 exit 退出。")

    while True:
        message = input("> ").strip()
        if message.lower() in {"exit", "quit"}:
            break

        matches = mapper.identify(message)
        result = {
            "status": "matched" if matches else "no_match",
            "companies": [match.to_dict() for match in matches],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

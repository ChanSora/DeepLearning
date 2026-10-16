#!/usr/bin/env python3
"""替换同目录聊天记录中的消息发送者，并合并所有 txt 文件。"""

from pathlib import Path
import re


TARGET_USER = "wxid_0txx9923tw0t22_afd6"
OUTPUT_NAME = "raw_data_chansora.txt"
MESSAGE_HEADER = re.compile(
    r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"(?P<sender>.+?)(?P<message_type>(?: {2,}\[[^\]\r\n]+\])*)$"
)


def replace_sender(line: str) -> str:
    """仅修改消息头的发送者，保留正文、时间和消息类型。"""
    match = MESSAGE_HEADER.fullmatch(line.rstrip("\r\n"))
    if match is None:
        return line

    sender = "chansora" if match["sender"] == TARGET_USER else "others"
    newline = line[len(line.rstrip("\r\n")):]
    return f'{match["timestamp"]} {sender}{match["message_type"]}{newline}'


def main() -> None:
    directory = Path(__file__).resolve().parent
    output_path = directory / OUTPUT_NAME
    input_paths = sorted(
        path for path in directory.glob("*.txt")
        if path.is_file() and path != output_path
    )
    if not input_paths:
        raise SystemExit("同目录下没有可合并的 txt 文件。")

    # utf-8-sig 同时兼容带 BOM 和不带 BOM 的 UTF-8 输入文件。
    with output_path.open("w", encoding="utf-8") as output:
        for index, input_path in enumerate(input_paths):
            if index:
                output.write("\n")
            last_line = ""
            with input_path.open("r", encoding="utf-8-sig") as source:
                for line in source:
                    output.write(replace_sender(line))
                    last_line = line
            if last_line and not last_line.endswith("\n"):
                output.write("\n")

    print(f"已合并 {len(input_paths)} 个文件：{output_path}")


if __name__ == "__main__":
    main()

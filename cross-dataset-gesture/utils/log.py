import os
import logging


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    OKEMPH = "\033[91m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


cs = [
    bcolors.HEADER,
    bcolors.OKBLUE,
    bcolors.OKCYAN,
    bcolors.OKGREEN,
    bcolors.WARNING,
    bcolors.FAIL,
    bcolors.ENDC,
    bcolors.BOLD,
    bcolors.UNDERLINE,
]


def init_logger(log_file_path: str):
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(log_file_path, mode="w")], force=True)


def logprint(content=""):
    pprint(content)
    log(content)


def pprint(content):
    print(f"{color('=', bcolors.OKCYAN)}  {content}")


def log(content):
    for c in cs:
        content = content.replace(c, "")

    if logging.getLogger().hasHandlers():
        logging.info(f"=\t{content}")


def logdivide(content="", padding=8, middle_padding=4):
    for c in cs:
        content = content.replace(c, "")

    terminal_size = os.get_terminal_size().columns
    content_len = len(content)
    minor_len = len("INFO:root: ")
    available = terminal_size - content_len - minor_len - padding - 2 * middle_padding - len("\t")
    available = max(0, available)

    dashes = "=" * (available // 2)
    spaces = f"{' ' * (padding)}"
    mid_spaces = " " * (middle_padding // 2)

    logging.info(f"{spaces}{dashes}{mid_spaces}{content}{mid_spaces}{dashes}{spaces}")


def color(content, code):
    return code + content + bcolors.ENDC


def header(content):
    return color(content, bcolors.HEADER)


def emph(content):
    return color(content, bcolors.OKGREEN)


def highlight(content):
    return color(content, bcolors.OKEMPH)


def setting(content):
    return color(content, bcolors.FAIL)


def printdivide(content="", padding=8, middle_padding=4):
    terminal_size = os.get_terminal_size().columns
    content_len = len(content)
    minor_len = 0
    available = terminal_size - content_len - minor_len - padding - 2 * middle_padding
    available = max(0, available)
    available = min(available, 300)

    dashes = f'{color("=", bcolors.OKCYAN)}' * (available // 2)
    mid_spaces = " " * (middle_padding // 2)

    print(f"{dashes}{mid_spaces}{content}{mid_spaces}{dashes}")


def divide(content="", padding=8, middle_padding=0):
    logdivide(content, padding, middle_padding)
    printdivide(content, padding, middle_padding)
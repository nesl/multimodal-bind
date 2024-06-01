import logging

def init_logger(train_log_file):
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(train_log_file, mode="w")], force=True)

def pprint(content):
    logging.info(f"===\t{content}")
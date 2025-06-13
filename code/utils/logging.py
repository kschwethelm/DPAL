import logging

def setup_logging(log_file):
    log_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove all existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.WARNING)

def ddp_logging(log, rank, force=False):
    if rank == 0 or force:
        logging.info(f"Rank {rank} - {log}")
    
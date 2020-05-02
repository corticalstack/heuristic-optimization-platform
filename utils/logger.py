import logging


def msg(l_type, m):
    """
    Log message and print to console if classified as info type
    """
    if l_type == logging.INFO:
        logging.info(m)
        print(m)
    elif l_type == logging.WARNING:
        logging.warning(m)
    elif l_type == logging.DEBUG:
        logging.debug(m)

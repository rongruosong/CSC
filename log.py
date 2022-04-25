# coding=utf-8
import time
import logging
from pathlib import Path


def setLogger(use_ch: bool = False):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Log等级总开关
    # 第二步，创建一个handler，用于写入日志文件
    rq = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    log_path = Path.cwd() / 'Logs'
    log_name = log_path / (rq + '.log')
    logfile = log_name
    log_path.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)  # 输出到file的log等级的开关

    # 第三步，定义handler的输出格式
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)

    # 第四步，将filehandler添加到logger里面
    logger.addHandler(fh)

    # console handler
    if use_ch:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


Logger = setLogger()
import logging
import os
from logging.handlers import RotatingFileHandler

# 日志配置常量
BASE_LOG_DIR = 'logs'
MAX_BYTES = 10 * 1024 * 1024  # 10MB
BACKUP_COUNT = 5


def setup_logging(service_name: str, level=logging.INFO):
    """
    集中配置日志系统，每个服务使用单独的日志文件。

    Args:
        service_name (str): 服务的名称（例如 'app' 或 'worker'），用于命名 logger 和日志文件。
        level (int): 日志级别。
    """
    # 确保日志目录存在
    os.makedirs(BASE_LOG_DIR, exist_ok=True)

    # 1. 确定日志文件路径
    log_file_name = f'{service_name}.log'
    log_file_path = os.path.join(BASE_LOG_DIR, log_file_name)

    # 2. 获取 logger 实例
    logger = logging.getLogger(service_name)
    logger.setLevel(level)

    # 避免重复添加 handlers（重要）
    if logger.hasHandlers():
        return logger

    # 3. 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 4. 控制台处理器 (StreamHandler) - 两个服务都保留控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 5. 文件处理器 (RotatingFileHandler) - 使用服务特定的路径
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    # 6. 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# 辅助函数，用于简化调用
def get_app_logger():
    return setup_logging(service_name='app')


def get_process_worker_logger():
    return setup_logging(service_name='process_worker')


def get_upload_worker_logger():
    return setup_logging(service_name='upload_worker')


class RequestLogger:
    """请求日志记录器（用于Flask）"""

    @staticmethod
    def log_request(request, response, duration: float = None):
        """
        记录HTTP请求日志

        Args:
            request: Flask request对象
            response: Flask response对象
            duration: 请求处理时间（秒）
        """
        logger = get_app_logger()

        # 构建日志消息
        msg_parts = [
            f"{request.method} {request.path}",
            f"status={response.status_code}",
        ]

        if duration is not None:
            msg_parts.append(f"duration={duration:.3f}s")

        # 添加查询参数
        if request.query_string:
            msg_parts.append(f"query={request.query_string.decode('utf-8')}")

        # 添加客户端IP
        client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
        msg_parts.append(f"ip={client_ip}")

        msg = " | ".join(msg_parts)

        # 根据状态码选择日志级别
        if response.status_code >= 500:
            logger.error(msg)
        elif response.status_code >= 400:
            logger.warning(msg)
        else:
            logger.info(msg)

    @staticmethod
    def log_error(request, error: Exception):
        """
        记录错误日志

        Args:
            request: Flask request对象
            error: 异常对象
        """
        logger = get_app_logger()
        logger.exception(
            f"请求错误 | {request.method} {request.path} | "
            f"error={type(error).__name__}: {str(error)}"
        )

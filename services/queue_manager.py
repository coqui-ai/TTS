"""
Redis 优先队列管理器 (仅用于队列操作，无状态存储)
"""

import json
import logging
import time
import uuid
from typing import Dict, Optional, Any

import redis
from redis import ConnectionPool

logger = logging.getLogger(__name__)


def _calculate_score(priority: int) -> float:
    """
    计算任务得分 (优先级)
    """
    timestamp = time.time()
    score = (6 - priority) * timestamp
    return score


class QueueManager:
    """Redis 队列管理器"""

    def __init__(self, redis_config: Dict[str, Any]):
        """
        初始化队列管理器

        Args:
            redis_config: Redis 配置字典
        """
        # 提取 Redis 连接参数
        host = redis_config['host']
        port = redis_config['port']
        db = redis_config['db']
        password = redis_config.get('password')

        max_connections = redis_config.get('max_connections', 10)

        # 1. 创建 Redis 连接池
        self.pool = ConnectionPool(
            host=host,
            port=port,
            db=db,
            password=password,
            max_connections=max_connections,
            decode_responses=True
        )

        self.redis_client = redis.Redis(connection_pool=self.pool)

        self.process_queue_key = redis_config['queue_key']

        self.upload_queue_key = redis_config['upload_queue_key']

        logger.info(
            f"QueueManager initialized with Connection Pool (Max={max_connections}): "
            f"Process Queue={self.process_queue_key}, Upload Queue={self.upload_queue_key}")

        try:
            self.redis_client.ping()
            logger.info("Redis connection successful.")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    def add_task(self, task_data: Dict[str, Any], priority: int = 3) -> str:
        """
        添加任务到处理队列

        Args:
            task_data: 任务数据 (包含 hook_url, text, etc.)
            priority: 优先级 (1-5)，1 为最高优先级

        Returns:
            str: 任务 ID
        """
        task_id = str(uuid.uuid4())

        task = {
            'task_id': task_id,
            'priority': priority,
            'created_at': time.time(),
            'data': task_data
        }

        score = _calculate_score(priority)

        # 添加到优先队列
        self.redis_client.zadd(self.process_queue_key, {json.dumps(task): score})

        logger.info(f"Task {task_id} added to process queue with priority {priority}, score {score}")

        return task_id

    def get_process_task(self) -> Optional[Dict[str, Any]]:
        """
        从处理队列 (ZSET) 中获取下一个任务

        Returns:
            Optional[Dict]: 任务数据，如果队列为空返回 None
        """
        # 使用 ZPOPMAX 获取得分最高的任务
        result = self.redis_client.zpopmax(self.process_queue_key, 1)

        if not result:
            return None

        task_json, score = result[0]
        task = json.loads(task_json)

        logger.info(f"Task {task.get('task_id', 'Unknown')} retrieved from process queue.")
        return task

    def push_upload_task(self, task_result: Dict[str, Any]):
        """
        将处理结果推送到上传队列 (List)

        Args:
            task_result: 任务处理结果 (包含 task_id, output_paths, hook_url等)
        """
        self.redis_client.lpush(self.upload_queue_key, json.dumps(task_result))
        logger.info(f"Task {task_result['task_id']} pushed to upload queue.")

    def get_upload_task(self, timeout: int = 5) -> Optional[Dict[str, Any]]:
        """
        从上传队列 (List) 中获取任务，阻塞等待

        Args:
            timeout: 阻塞等待时间 (秒)

        Returns:
            Optional[Dict]: 任务数据，如果超时返回 None
        """
        # 使用 BRPOP 阻塞弹出
        result = self.redis_client.brpop(self.upload_queue_key, timeout)

        if not result:
            return None

        # BRPOP 返回 (key, value)
        task_json = result[1]
        task = json.loads(task_json)

        logger.debug(f"Task {task.get('task_id', 'Unknown')} retrieved from upload queue.")
        return task

    def get_process_queue_stats(self) -> Dict[str, Any]:
        """
        获取处理队列统计信息
        """
        queued_count = self.redis_client.zcard(self.process_queue_key)
        upload_count = self.redis_client.llen(self.upload_queue_key)

        return {
            'process_queued': queued_count,
            'upload_queued': upload_count,
            'timestamp': time.time()
        }

    def delete_process_task(self, task_id: str) -> bool:
        """
        根据 task_id 从处理队列中删除任务。

        Args:
            task_id: 要删除的任务的 ID。

        Returns:
            bool: 如果任务被成功删除返回 True，否则返回 False。
        """
        deleted = False

        # 使用 ZSCAN 迭代 ZSET 的成员，以避免一次性加载所有数据
        # cursor = 0
        # while True:
        #     # count 参数用于每次扫描返回的元素数量提示
        #     # scan_result 格式: (新的 cursor, [member1, member2, ...])
        #     cursor, members_with_scores = self.redis_client.zscan(self.process_queue_key, cursor=cursor, count=100)

        #     # ZSCAN 返回的 members_with_scores 是一个元组列表 [(member1, score1), (member2, score2), ...]
        #     members = [item[0] for item in members_with_scores]
        #
        #     for member_json in members:
        #         try:
        #             task = json.loads(member_json)
        #             if task.get('task_id') == task_id:
        #                 # 找到匹配项，使用 ZREM 删除
        #                 # zrem 返回被移除的元素数量
        #                 removed_count = self.redis_client.zrem(self.process_queue_key, member_json)
        #
        #                 if removed_count > 0:
        #                     logger.warning(f"Task {task_id} successfully deleted from process queue.")
        #                     deleted = True
        #                     # 由于只需要删除一个，找到后可以退出循环
        #                     return True
        #         except json.JSONDecodeError:
        #             # 忽略非法的 JSON 成员
        #             logger.error(f"Failed to decode ZSET member: {member_json}")
        #             pass

        #     if cursor == 0:
        #         break

        # --- 考虑到 ZSCAN 循环的复杂性，如果队列规模可控，可以先使用 ZRANGEBYSCORE 或 ZRANGE 获取所有成员进行查找 ---
        # 另一种查找方式: 获取所有成员 (member-only)
        # 如果队列规模太大 (如超过数十万)，可能会有性能问题。请根据实际情况选择 ZSCAN 或 ZRANGE。

        members = self.redis_client.zrange(self.process_queue_key, 0, -1)

        for member_json in members:
            try:
                task = json.loads(member_json)
                if task.get('task_id') == task_id:
                    # 找到匹配项，使用 ZREM 删除
                    removed_count = self.redis_client.zrem(self.process_queue_key, member_json)

                    if removed_count > 0:
                        logger.warning(f"Task {task_id} successfully deleted from process queue.")
                        return True  # 删除成功

            except json.JSONDecodeError:
                logger.error(f"Failed to decode ZSET member: {member_json}")

        logger.info(f"Task {task_id} not found in process queue.")
        return False

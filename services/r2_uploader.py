"""
Cloudflare R2 文件上传服务
"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class R2Uploader:
    """Cloudflare R2 上传器"""

    def __init__(self, r2_config: Dict[str, Any]):
        """
        初始化 R2 上传器

        Args:
            r2_config: R2 配置字典，包含 'bucket_name', 'endpoint_url', 'access_key_id', 'secret_access_key', 'public_url' 等

        Raises:
            ValueError: 配置无效
            ClientError: 凭证或 Bucket 访问失败
        """
        self.config = r2_config
        self.bucket_name = r2_config['bucket_name']
        self.public_url = r2_config.get('public_url', '').rstrip('/')

        # 验证必需配置
        required_keys = ['bucket_name', 'endpoint_url', 'access_key_id', 'secret_access_key']
        missing = [k for k in required_keys if k not in r2_config]
        if missing:
            raise ValueError(f"Missing required R2 config keys: {missing}")

        # 初始化 S3 客户端（R2 兼容 S3 API）
        # 添加重试配置和签名版本（R2 要求 s3v4）
        client_config = Config(
            signature_version='s3v4',  # R2 必需的签名版本
            retries={
                'max_attempts': 3,
                'mode': 'standard'
            },
            connect_timeout=10,
            read_timeout=10
        )

        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=r2_config['endpoint_url'],
                aws_access_key_id=r2_config['access_key_id'],
                aws_secret_access_key=r2_config['secret_access_key'],
                region_name='auto',  # R2 使用 'auto' 作为区域
                config=client_config
            )
        except NoCredentialsError:
            raise ValueError("Invalid AWS credentials (Access Key/Secret Key)")

        # 验证凭证和 Bucket 访问（测试 head_bucket 以检查权限）
        self._validate_bucket_access()

        logger.info(f"R2Uploader initialized for bucket: {self.bucket_name}")

    def _validate_bucket_access(self):
        """验证 Bucket 访问权限（测试 PutObject 权限模拟）"""
        try:
            # 先检查 Bucket 存在和基本访问
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.debug(f"Bucket '{self.bucket_name}' access confirmed")

            # 简单测试：尝试列出对象（如果权限不足，会早抛 AccessDenied）
            self.s3_client.list_objects_v2(Bucket=self.bucket_name, MaxKeys=0)

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            if error_code == 'AccessDenied':
                raise ClientError(
                    e.response,
                    "AccessDenied: Check API Token permissions (requires 'Object Read & Write' for bucket). "
                    f"Ensure token is bound to bucket '{self.bucket_name}'. Details: {error_msg}"
                )
            elif error_code == 'NoSuchBucket':
                raise ClientError(
                    e.response,
                    f"Bucket '{self.bucket_name}' does not exist or is not accessible."
                )
            else:
                raise

    def upload_file(
            self,
            file_path: str,
            object_key: Optional[str] = None,
            metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """
        上传文件到 R2

        Args:
            file_path: 本地文件路径
            object_key: R2 对象键（路径），如果为 None 则使用文件名
            metadata: 文件元数据（注意：R2 支持基本 Metadata，但不支持 Tagging）

        Returns:
            str: 文件的公开 URL

        Raises:
            FileNotFoundError: 文件不存在
            ClientError: 上传失败（包含详细错误信息）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # 如果没有指定 object_key，使用文件名
        if object_key is None:
            object_key = os.path.basename(file_path)

        # 确保 object_key 不为空且无前导 /
        object_key = object_key.lstrip('/')

        try:
            # 准备上传参数（避免不支持参数，如 ACL）
            extra_args = {}

            # 设置内容类型
            content_type = self._get_content_type(file_path)
            if content_type:
                extra_args['ContentType'] = content_type

            # 设置元数据（R2 支持）
            if metadata:
                extra_args['Metadata'] = metadata

            # 上传文件
            logger.info(f"Uploading {file_path} to R2 bucket '{self.bucket_name}' as '{object_key}'")

            self.s3_client.upload_file(
                file_path,
                self.bucket_name,
                object_key,
                ExtraArgs=extra_args
            )

            # 生成公开 URL
            file_url = f"{self.public_url}/{object_key}" if self.public_url else None

            logger.info(f"File uploaded successfully: {file_url or object_key}")

            return file_url or object_key

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(
                f"Failed to upload {file_path} to R2: Code={error_code}, Message={error_msg}. "
                f"Check: 1) API Token has 'Object Read & Write' permission bound to bucket '{self.bucket_name}'. "
                f"2) No unsupported params (e.g., ACL, Tagging). 3) Endpoint/Region correct."
            )
            raise ClientError(e.response, f"Upload failed: {error_msg}")
        except Exception as e:
            logger.error(f"Unexpected error uploading {file_path}: {str(e)}")
            raise

    def upload_files(
            self,
            file_paths: list,
            prefix: str = '',
            metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        批量上传文件到 R2

        Args:
            file_paths: 本地文件路径列表
            prefix: R2 路径前缀（会自动添加 / 如果需要）
            metadata: 文件元数据

        Returns:
            Dict[str, str]: 文件路径到 URL 的映射（失败为 None）
        """
        results = {}
        prefix = prefix.lstrip('/')  # 清理前导 /

        for file_path in file_paths:
            try:
                # 生成 object_key
                filename = os.path.basename(file_path)
                object_key = f"{prefix}/{filename}" if prefix else filename

                # 上传文件
                file_url = self.upload_file(file_path, object_key, metadata)
                results[file_path] = file_url

            except Exception as e:
                logger.error(f"Failed to upload {file_path}: {str(e)}")
                results[file_path] = None

        successful = sum(1 for url in results.values() if url is not None)
        logger.info(f"Batch upload completed: {successful}/{len(file_paths)} files uploaded to '{self.bucket_name}'")

        return results

    def delete_file(self, object_key: str):
        """
        从 R2 删除文件

        Args:
            object_key: R2 对象键

        Raises:
            ClientError: 删除失败
        """
        object_key = object_key.lstrip('/')
        try:
            logger.info(f"Deleting '{object_key}' from R2 bucket '{self.bucket_name}'")

            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=object_key
            )

            logger.info(f"File deleted successfully: {object_key}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to delete {object_key}: Code={error_code}, Message={error_msg}")
            raise

    def delete_files(self, object_keys: list):
        """
        批量删除文件

        Args:
            object_keys: R2 对象键列表
        """
        if not object_keys:
            return

        # 清理键
        cleaned_keys = [k.lstrip('/') for k in object_keys if k.strip()]

        try:
            # 准备删除对象列表
            delete_objects = [{'Key': key} for key in cleaned_keys]

            logger.info(f"Deleting {len(cleaned_keys)} files from R2 bucket '{self.bucket_name}'")

            response = self.s3_client.delete_objects(
                Bucket=self.bucket_name,
                Delete={'Objects': delete_objects}
            )

            deleted_count = len(response.get('Deleted', []))
            errors = response.get('Errors', [])
            if errors:
                logger.warning(f"Batch delete errors: {errors}")

            logger.info(f"Batch delete completed: {deleted_count}/{len(cleaned_keys)} files deleted")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to batch delete files: Code={error_code}, Message={error_msg}")
            raise

    def file_exists(self, object_key: str) -> bool:
        """
        检查文件是否存在

        Args:
            object_key: R2 对象键

        Returns:
            bool: 文件是否存在
        """
        object_key = object_key.lstrip('/')
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=object_key
            )
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                error_msg = e.response['Error']['Message']
                logger.error(f"Error checking existence of {object_key}: {error_msg}")
                raise

    def get_file_url(self, object_key: str) -> str:
        """
        获取文件的公开 URL

        Args:
            object_key: R2 对象键

        Returns:
            str: 文件 URL
        """
        object_key = object_key.lstrip('/')
        return f"{self.public_url}/{object_key}" if self.public_url else object_key

    def _get_content_type(self, file_path: str) -> Optional[str]:
        """
        根据文件扩展名获取内容类型

        Args:
            file_path: 文件路径

        Returns:
            Optional[str]: 内容类型
        """
        extension = Path(file_path).suffix.lower()

        content_types = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.mp4': 'video/mp4',  # 修正为 video/mp4（原为 audio/mp4，仅音频用 m4a）
            '.ogg': 'audio/ogg',
            '.flac': 'audio/flac',
            '.aac': 'audio/aac',
            '.m4a': 'audio/mp4',
            '.srt': 'text/plain',
            '.txt': 'text/plain',
            '.json': 'application/json',
            '.zip': 'application/zip',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
        }

        return content_types.get(extension, 'application/octet-stream')  # 默认二进制

    def list_files(self, prefix: str = '', max_keys: int = 1000) -> list:
        """
        列出 R2 中的文件

        Args:
            prefix: 路径前缀
            max_keys: 最大返回数量

        Returns:
            list: 文件列表 [{'key': str, 'size': int, 'last_modified': str, 'url': str}]
        """
        prefix = prefix.lstrip('/')
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix,
                MaxKeys=max_keys
            )

            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'url': self.get_file_url(obj['Key'])
                    })

            logger.info(f"Listed {len(files)} files with prefix '{prefix}' in '{self.bucket_name}'")

            return files

        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_msg = e.response['Error']['Message']
            logger.error(f"Failed to list files: Code={error_code}, Message={error_msg}")
            raise

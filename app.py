#!/usr/bin/env python3
"""
Flask API Server for Voice Clone Service
支持单文件和批量处理，使用 Redis 优先队列，上传至 R2
"""

import os
import time
from functools import wraps

import jwt
import yaml
from flask import Flask, request, jsonify, g

from services.logger import get_app_logger, RequestLogger
from services.queue_manager import QueueManager
from services.r2_uploader import R2Uploader
from services.voice_processor import VoiceProcessor

logger = get_app_logger()


# 加载配置
def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()


def token_required(f):
    """JWT 认证装饰器"""

    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # 尝试从 Authorization: Bearer <token> 头中获取 token
        auth_header = request.headers.get('Authorization')

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]

        if not token:
            return jsonify({
                'message': 'Token is missing!',
                'error': 'Missing Authorization header or Bearer token'
            }), 401

        try:
            # 解码 Token
            jwt.decode(
                token,
                config['jwt']['secret_key'],
                algorithms=[config['jwt']['algorithm']]
            )
        except jwt.ExpiredSignatureError:
            return jsonify({
                'message': 'Token is expired!'
            }), 401
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid Token attempt: {e}")
            return jsonify({
                'message': 'Token is invalid!',
                'error': str(e)
            }), 401

        return f(*args, **kwargs)

    return decorated


# 初始化 Flask
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# 初始化服务
queue_manager = QueueManager(config['redis'])
voice_processor = VoiceProcessor(config['voice_clone'])
r2_uploader = R2Uploader(config['r2'])


# 请求开始时间
@app.before_request
def before_request():
    g.start_time = time.time()


# 设置响应的Content-Type和记录请求日志
@app.after_request
def after_request(response):
    if response.content_type and 'application/json' in response.content_type:
        response.headers['Content-Type'] = 'application/json; charset=utf-8'

    # 确保响应数据使用UTF-8编码
    response.charset = 'utf-8'

    # 记录请求日志
    if hasattr(g, 'start_time'):
        duration = time.time() - g.start_time
        RequestLogger.log_request(request, response, duration)

    return response


@app.route('/generate', methods=['POST'])
@token_required
def clone_single():
    """
    单文件语音克隆

    Body (JSON):
    {
      "spk_audio_prompt": "https://tts.luckyshort.net/seg_001.wav", // 原语音
      "text": "What good is a backwards method like that?", // clone文本内容
      "priority": 3, //默认优先级为3，1为优先级最低，5为优先级最高
      "hook_url": "https://api.vibevibe.vip", // 回调地址
      "params": { // 可选参数
        "language": "EN"
      }
    }
    """
    try:
        data = request.json
        text = data.get('text')
        spk_audio_prompt = data.get('spk_audio_prompt')
        priority = data.get('priority', 3)
        hook_url = data.get('hook_url')
        params = data.get('params', {})
        language = params.get('language', 'en')  # 默认使用 en（小写格式，直接传递）

        # 验证参数
        if not text:
            return jsonify({'error': 'Missing required field: text'}), 400

        if not hook_url:
            return jsonify({'error': 'Missing required field: hook_url'}), 400

        if not spk_audio_prompt:
            return jsonify({'error': 'Missing required field: spk_audio_prompt'}), 400

        if priority not in range(1, 6):
            return jsonify({'error': 'Priority must be between 1 and 5'}), 400

        if language not in config['voice_clone']['supported_languages']:
            return jsonify({'error': f'Unsupported language: {language}'}), 400

        # 创建任务
        task = {
            'text': text,
            'language': language,
            'spk_audio_prompt': spk_audio_prompt,
            'hook_url': hook_url,
            'priority': priority
        }

        # 添加到队列
        task_id = queue_manager.add_task(task, priority)

        logger.info(f"Created task: {task_id}, priority: {priority}")

        return jsonify({
            'task_uuid': task_id,
            'status': 'queued',
            'message': 'Task added to queue successfully'
        }), 201

    except Exception as e:
        logger.error(f"Error in clone_single: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/tasks/<task_id>/cancel', methods=['DELETE'])
@token_required
def cancel_task(task_id: str):
    try:
        if not task_id:
            return jsonify({'error': 'Missing required field: task_id'}), 400

        # 删除任务
        if queue_manager.delete_process_task(task_id):
            logger.info(f"Canceled  task: {task_id}")

        return jsonify({
            'task_uuid': task_id,
            'status': 'canceled',
            'message': 'Task canceled successfully'
        }), 200

    except Exception as e:
        logger.error(f"Error in clone_single: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(413)
def request_entity_too_large(error):
    """文件过大错误处理"""
    return jsonify({'error': 'File too large. Maximum size is 100MB'}), 413


@app.errorhandler(500)
def internal_error(error):
    """服务器内部错误处理"""
    logger.error(f"Internal error: {str(error)}")
    RequestLogger.log_error(request, error)
    return jsonify({'error': 'Internal server error'}), 500


def main():
    """启动服务"""
    logger.info("Starting Voice Clone API Server...")
    logger.info("Config loaded from: config.yaml")
    logger.info(f"Output directory: {config['voice_clone']['output_dir']}")

    # 创建必要的目录
    os.makedirs(config['voice_clone']['output_dir'], exist_ok=True)
    os.makedirs(config['voice_clone']['temp_dir'], exist_ok=True)

    # 启动 Flask
    app.run(
        host=config['app']['host'],
        port=config['app']['port'],
        debug=config['app']['debug']
    )


if __name__ == '__main__':
    main()

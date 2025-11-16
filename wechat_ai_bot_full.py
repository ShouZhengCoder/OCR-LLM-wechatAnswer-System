#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微信OCR自动回复机器人 - 完整版
功能：基于颜色识别和OCR技术的微信自动回复系统
依赖：pip install pillow pytesseract pyautogui pygetwindow openai pyperclip numpy scipy
"""

import os
import re
import time
import hashlib
import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pyautogui
import pygetwindow as gw
import pytesseract
import pyperclip
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageOps
from openai import OpenAI

# ==================== 配置区 ====================
CONFIG = {
    # API配置
    'API_KEY': 'YOUR_DEEPSEEK_API_KEY',
    'API_BASE': 'https://api.deepseek.com',
    'MODEL': 'deepseek-chat',

    # OCR配置
    'TESSERACT_PATH': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'OCR_LANG': 'chi_sim+eng',  # 中文简体+英文
    'OCR_PSM': 4,  # Page Segmentation Mode: 4=假设一列文本
    'OCR_OEM': 3,  # OCR Engine Mode: 3=默认（基于LSTM）

    # 运行参数
    'CHECK_INTERVAL': 3,  # 检测间隔（秒）
    'AUTO_REPLY_ENABLED': True,  # 自动回复开关
    'WORK_HOURS': None,  # 工作时间限制，如 {'start': 9, 'end': 18}
    'REPLY_COOLDOWN': 10,  # 回复后冷却时间（秒）
    'SKIP_CYCLES_AFTER_SEND': 3,  # 发送后跳过检测周期数

    # 图像处理参数
    'FILTER_GREEN_BUBBLES': True,  # 是否屏蔽绿色气泡
    'GREEN_DETECTION_PARAMS': {
        'g_min': 170,  # G通道最小值
        'r_max': 190,  # R通道最大值
        'diff_min': 40,  # G-R最小差值
        'dilation_iterations': 20  # 膨胀迭代次数
    },
    'IMAGE_SCALE_FACTOR': 1.7,  # OCR前图像放大倍数
    'CHANGE_DETECTION_MIN_AREA': 600,  # 最小变化面积（像素）
    'BBOX_MARGIN': 20,  # 边界框扩展边距

    # 消息过滤
    'MESSAGE_MIN_LENGTH': 2,  # 最小消息长度
    'MESSAGE_HISTORY_SIZE': 50,  # 消息历史记录大小
    'CONTEXT_HISTORY_SIZE': 12,  # 对话上下文大小
    'NOISE_KEYWORDS': [
        '复制', '转发', '撤回', '编辑', '收藏',
        '消息', '表情', '图片', '视频', '文件',
        '链接', '语音', '已读', '未读'
    ],

    # 调试选项
    'DEBUG_MODE': False,  # 调试模式
    'SAVE_DEBUG_IMAGES': False,  # 是否保存调试图像
    'DEBUG_IMAGE_DIR': './debug_images',  # 调试图像保存目录
    'LOG_LEVEL': 'INFO',  # 日志级别: DEBUG, INFO, WARNING, ERROR
}

# ==================== 日志配置 ====================
def setup_logger(level: str = 'INFO') -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger('WeChatAIBot')
    logger.setLevel(getattr(logging, level.upper()))

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    return logger

logger = setup_logger(CONFIG['LOG_LEVEL'])

# ==================== DeepSeek AI 客户端 ====================
class DeepSeekAI:
    """DeepSeek API调用封装"""

    def __init__(self, api_key: str, base_url: str, model: str):
        """
        初始化DeepSeek AI客户端

        Args:
            api_key: DeepSeek API密钥
            base_url: API基础URL
            model: 使用的模型名称
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_prompt = """你是一个智能微信自动回复助手。请根据收到的消息生成合适的回复。

要求：
1. 回复要自然、友好、简洁（50字以内）
2. 根据消息内容判断意图，给出恰当回应
3. 如果是问候，友好回应
4. 如果是询问，简要回答或表示稍后处理
5. 如果是闲聊，自然对话
6. 保持礼貌和专业性
7. 不要使用表情符号，除非对方使用了"""

        logger.info(f"DeepSeek AI 初始化完成，模型: {model}")

    def generate_reply(self, message: str, context: Optional[str] = None) -> str:
        """
        生成回复内容

        Args:
            message: 收到的消息
            context: 对话上下文（可选）

        Returns:
            生成的回复文本
        """
        try:
            messages = [{"role": "system", "content": self.system_prompt}]

            if context:
                messages.append({
                    "role": "user",
                    "content": f"对话历史：\n{context}"
                })

            messages.append({
                "role": "user",
                "content": f"收到消息：{message}\n\n请生成回复："
            })

            logger.debug(f"发送API请求，消息长度: {len(message)}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=100,
                temperature=0.7
            )

            reply = response.choices[0].message.content.strip()
            logger.debug(f"AI回复: {reply}")

            return reply

        except Exception as e:
            logger.error(f"AI生成回复失败: {e}")
            return "收到你的消息，稍后回复~"

# ==================== OCR识别模块 ====================
class WeChatOCR:
    """微信消息OCR识别"""

    def __init__(self, tesseract_path: str, lang: str, config: Dict[str, Any]):
        """
        初始化OCR识别器

        Args:
            tesseract_path: Tesseract可执行文件路径
            lang: OCR语言设置
            config: 配置字典
        """
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            logger.info(f"Tesseract路径已设置: {tesseract_path}")
        else:
            logger.warning(f"Tesseract路径不存在: {tesseract_path}")

        self.lang = lang
        self.config = config
        self.last_screenshot: Optional[Image.Image] = None

        # 创建调试图像目录
        if config['SAVE_DEBUG_IMAGES']:
            os.makedirs(config['DEBUG_IMAGE_DIR'], exist_ok=True)
            logger.info(f"调试图像目录: {config['DEBUG_IMAGE_DIR']}")

    def find_wechat_window(self):
        """
        查找微信窗口

        Returns:
            微信窗口对象，未找到返回None
        """
        windows = gw.getWindowsWithTitle('微信')
        if windows:
            logger.debug(f"找到微信窗口: {windows[0].title}")
            return windows[0]
        else:
            logger.debug("未找到微信窗口")
            return None

    def capture_chat_area(self, window) -> Optional[Image.Image]:
        """
        截取聊天消息区域

        Args:
            window: 微信窗口对象

        Returns:
            截图图像，失败返回None
        """
        try:
            # 激活窗口
            window.activate()
            time.sleep(0.1)

            # 计算聊天区域坐标
            # 跳过左侧联系人列表（35%）和顶部标题栏（80px）
            x = window.left + int(window.width * 0.35)
            y = window.top + 80
            width = int(window.width * 0.62)  # 聊天区域宽度
            height = int(window.height * 0.70)  # 排除底部输入框

            logger.debug(f"截图区域: x={x}, y={y}, w={width}, h={height}")

            screenshot = pyautogui.screenshot(region=(x, y, width, height))

            # 保存调试图像
            if self.config['SAVE_DEBUG_IMAGES']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = os.path.join(
                    self.config['DEBUG_IMAGE_DIR'],
                    f'screenshot_{timestamp}.png'
                )
                screenshot.save(debug_path)
                logger.debug(f"保存截图: {debug_path}")

            return screenshot

        except Exception as e:
            logger.error(f"截图失败: {e}")
            return None

    def get_change_bbox(self, current_screenshot: Image.Image,
                       min_area: int = 600) -> Optional[Tuple[int, int, int, int]]:
        """
        检测画面变化区域

        Args:
            current_screenshot: 当前截图
            min_area: 最小变化面积

        Returns:
            变化区域边界框(x0, y0, x1, y1)，无变化返回None
        """
        if self.last_screenshot is None:
            self.last_screenshot = current_screenshot
            logger.debug("首次截图，保存为基准")
            return None

        try:
            # 计算图像差异
            diff = ImageChops.difference(current_screenshot, self.last_screenshot)
            bbox = diff.getbbox()

            # 更新基准截图
            self.last_screenshot = current_screenshot

            if not bbox:
                logger.debug("无画面变化")
                return None

            # 计算变化面积
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            if area >= min_area:
                logger.debug(f"检测到变化区域: {bbox}, 面积: {area}px²")
                return bbox
            else:
                logger.debug(f"变化面积过小: {area}px² < {min_area}px²")
                return None

        except Exception as e:
            logger.error(f"变化检测失败: {e}")
            self.last_screenshot = current_screenshot
            return None

    @staticmethod
    def expand_bbox(bbox: Tuple[int, int, int, int],
                    image_size: Tuple[int, int],
                    margin: int = 20) -> Tuple[int, int, int, int]:
        """
        扩展边界框

        Args:
            bbox: 原始边界框
            image_size: 图像尺寸(width, height)
            margin: 扩展边距

        Returns:
            扩展后的边界框
        """
        x0, y0, x1, y1 = bbox
        w, h = image_size

        return (
            max(0, x0 - margin),
            max(0, y0 - margin),
            min(w, x1 + margin),
            min(h, y1 + margin)
        )

    def mask_green_bubbles(self, image: Image.Image) -> Image.Image:
        """
        屏蔽绿色气泡（自己的消息）

        Args:
            image: 原始图像

        Returns:
            屏蔽绿色后的图像
        """
        try:
            img_array = np.array(image)
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

            # 获取绿色检测参数
            params = self.config['GREEN_DETECTION_PARAMS']

            # 检测绿色区域：G通道高、R通道低、G明显大于R
            green_mask = (
                (g > params['g_min']) &
                (r < params['r_max']) &
                ((g - r) > params['diff_min'])
            )

            green_pixels_before = np.sum(green_mask)

            # 膨胀绿色区域，覆盖气泡上的黑色文字
            try:
                from scipy.ndimage import binary_dilation
                green_mask_dilated = binary_dilation(
                    green_mask,
                    iterations=params['dilation_iterations']
                )
            except ImportError:
                logger.warning("scipy未安装，无法进行膨胀操作，准确率可能降低")
                logger.info("建议执行: pip install scipy")
                green_mask_dilated = green_mask

            green_pixels_after = np.sum(green_mask_dilated)

            # 将绿色区域填充为背景色
            img_array[green_mask_dilated] = [245, 245, 245]

            logger.debug(
                f"绿色屏蔽: {green_pixels_before}px → "
                f"{green_pixels_after}px (膨胀{params['dilation_iterations']}次)"
            )

            result = Image.fromarray(img_array)

            # 保存调试图像
            if self.config['SAVE_DEBUG_IMAGES']:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = os.path.join(
                    self.config['DEBUG_IMAGE_DIR'],
                    f'masked_{timestamp}.png'
                )
                result.save(debug_path)
                logger.debug(f"保存屏蔽后图像: {debug_path}")

            return result

        except Exception as e:
            logger.error(f"绿色屏蔽失败: {e}")
            return image

    def keep_text_regions(self, image: Image.Image) -> Image.Image:
        """
        保留文字区域，其他区域填充背景色（增强OCR识别）

        Args:
            image: 输入图像

        Returns:
            处理后的图像
        """
        try:
            # 转为灰度并反转
            inverted = ImageOps.invert(image.convert('L'))

            # 轻微模糊
            blurred = inverted.filter(ImageFilter.BoxBlur(1))

            # 自适应阈值
            arr = np.array(blurred)
            threshold = np.percentile(arr, 70)

            # 二值化
            binary = blurred.point(lambda p: 255 if p > threshold else 0)

            # 膨胀文字区域
            dilated = binary.filter(ImageFilter.MaxFilter(9))

            # 创建背景
            background = Image.new('RGB', image.size, (245, 245, 245))

            # 合成：保留文字区域，其他用背景填充
            result = Image.composite(image, background, dilated)

            logger.debug("文本区域提取完成")

            return result

        except Exception as e:
            logger.error(f"文本区域提取失败: {e}")
            return image

    def ocr_recognize(self, image: Image.Image, filter_green: bool = True) -> str:
        """
        OCR识别文本

        Args:
            image: 待识别图像
            filter_green: 是否过滤绿色气泡

        Returns:
            识别到的文本
        """
        try:
            # 屏蔽绿色气泡
            if filter_green:
                image = self.mask_green_bubbles(image)

            # 保留文字区域
            image = self.keep_text_regions(image)

            # 放大图像以提高识别率
            scale = self.config['IMAGE_SCALE_FACTOR']
            w, h = image.size
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            image = image.resize(new_size, Image.Resampling.BILINEAR)

            # 转灰度并增强对比度
            gray = ImageOps.autocontrast(image.convert('L'))

            # OCR配置
            custom_config = (
                f'--oem {self.config["OCR_OEM"]} '
                f'--psm {self.config["OCR_PSM"]}'
            )

            # 执行OCR
            text = pytesseract.image_to_string(
                gray,
                lang=self.lang,
                config=custom_config
            )

            result = text.strip()

            if result:
                logger.debug(f"OCR识别成功，文本长度: {len(result)}")
            else:
                logger.debug("OCR未识别到文本")

            return result

        except Exception as e:
            logger.error(f"OCR识别失败: {e}")
            return ""

    def extract_latest_message(self, full_text: str) -> Optional[str]:
        """
        从OCR文本中提取最新消息

        Args:
            full_text: OCR识别的完整文本

        Returns:
            提取的最新消息，无有效消息返回None
        """
        if not full_text:
            return None

        # 分割为行
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]

        if not lines:
            return None

        # 过滤噪声
        noise_keywords = self.config['NOISE_KEYWORDS']
        min_length = self.config['MESSAGE_MIN_LENGTH']

        filtered_lines = []
        for line in lines:
            # 过滤时间戳
            if re.match(r'^\d{1,2}:\d{2}$', line):
                continue

            # 过滤过短的行
            if len(line) < min_length:
                continue

            # 过滤噪声关键词
            if any(keyword in line for keyword in noise_keywords):
                continue

            filtered_lines.append(line)

        # 取最后2行作为最新消息
        if filtered_lines:
            message = ' '.join(filtered_lines[-2:])
            logger.debug(f"提取消息: {message}")
            return message

        return None

# ==================== UI自动化模块 ====================
class WeChatUI:
    """微信UI自动化操作"""

    @staticmethod
    def send_message(window, text: str) -> bool:
        """
        发送消息

        Args:
            window: 微信窗口对象
            text: 要发送的文本

        Returns:
            是否发送成功
        """
        try:
            # 计算输入框位置
            input_x = window.left + int(window.width * 0.65)
            input_y = window.top + window.height - 120

            # 点击输入框
            pyautogui.click(input_x, input_y)
            time.sleep(0.15)

            # 清空输入框
            pyautogui.hotkey('ctrl', 'a')
            time.sleep(0.05)

            # 使用剪贴板输入文本（支持中文）
            pyperclip.copy(text)
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.2)

            # 发送
            pyautogui.press('enter')

            logger.debug(f"消息已发送: {text[:50]}...")

            return True

        except Exception as e:
            logger.error(f"消息发送失败: {e}")
            return False

# ==================== 主机器人类 ====================
class WeChatAIBot:
    """微信AI自动回复机器人"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化机器人

        Args:
            config: 配置字典
        """
        self.config = config

        # 初始化AI客户端
        self.ai = DeepSeekAI(
            config['API_KEY'],
            config['API_BASE'],
            config['MODEL']
        )

        # 初始化OCR识别器
        self.ocr = WeChatOCR(
            config['TESSERACT_PATH'],
            config['OCR_LANG'],
            config
        )

        # 消息历史（用于去重）
        self.message_history: set = set()

        # 对话上下文
        self.conversation_context: List[str] = []

        # 最后回复时间和内容
        self.last_reply_time: float = 0
        self.last_reply_content: str = ""

        # 跳过检测周期计数器
        self.skip_cycles: int = 0

        logger.info("微信AI机器人初始化完成")

    def get_message_hash(self, message: str) -> str:
        """
        生成消息哈希值

        Args:
            message: 消息文本

        Returns:
            MD5哈希值
        """
        return hashlib.md5(message.encode()).hexdigest()

    def is_duplicate_message(self, message: str) -> bool:
        """
        检查是否为重复消息

        Args:
            message: 消息文本

        Returns:
            是否重复
        """
        msg_hash = self.get_message_hash(message)

        if msg_hash in self.message_history:
            return True

        # 添加到历史
        self.message_history.add(msg_hash)

        # 限制历史大小
        max_size = self.config['MESSAGE_HISTORY_SIZE']
        if len(self.message_history) > max_size:
            self.message_history.pop()

        return False

    def is_recent_self_reply(self, message: str) -> bool:
        """
        检查是否为最近自己的回复

        Args:
            message: 消息文本

        Returns:
            是否为自己的回复
        """
        current_time = time.time()
        cooldown = self.config['REPLY_COOLDOWN']

        # 在冷却时间内
        if current_time - self.last_reply_time < cooldown:
            # 完全匹配
            if message.strip() == self.last_reply_content.strip():
                return True

            # 关键词匹配
            self_reply_keywords = ['回复', '稍后', '收到', '系统']
            if any(keyword in message for keyword in self_reply_keywords):
                return True

            # 模糊匹配（相似度>85%）
            similarity = SequenceMatcher(
                None,
                message.strip(),
                self.last_reply_content.strip()
            ).ratio()

            if similarity > 0.85:
                logger.debug(f"检测到高相似度消息: {similarity:.2%}")
                return True

        return False

    def should_auto_reply(self) -> bool:
        """
        判断是否应该自动回复

        Returns:
            是否应该回复
        """
        # 检查总开关
        if not self.config['AUTO_REPLY_ENABLED']:
            return False

        # 检查工作时间限制
        work_hours = self.config.get('WORK_HOURS')
        if work_hours:
            current_hour = datetime.now().hour
            if work_hours['start'] <= current_hour < work_hours['end']:
                logger.debug(f"当前时间{current_hour}在工作时间内，不自动回复")
                return False

        return True

    def update_context(self, message: str, reply: str):
        """
        更新对话上下文

        Args:
            message: 用户消息
            reply: 机器人回复
        """
        self.conversation_context.append(f"用户: {message}")
        self.conversation_context.append(f"我: {reply}")

        # 限制上下文大小
        max_size = self.config['CONTEXT_HISTORY_SIZE']
        if len(self.conversation_context) > max_size:
            self.conversation_context = self.conversation_context[-max_size:]

        # 更新最后回复信息
        self.last_reply_time = time.time()
        self.last_reply_content = reply

    def run(self):
        """运行机器人主循环"""
        logger.info("=" * 60)
        logger.info("微信AI自动回复机器人启动")
        logger.info(f"AI模型: {self.config['MODEL']}")
        logger.info(f"检测间隔: {self.config['CHECK_INTERVAL']}秒")
        logger.info(f"调试模式: {'开启' if self.config['DEBUG_MODE'] else '关闭'}")
        logger.info("=" * 60)
        logger.info("\n请确保：")
        logger.info("1. 微信已登录且窗口可见")
        logger.info("2. 打开需要自动回复的聊天窗口")
        logger.info("3. DeepSeek API Key已配置")
        logger.info("\n按 Ctrl+C 停止运行\n")

        while True:
            try:
                # 强制跳过机制
                if self.skip_cycles > 0:
                    self.skip_cycles -= 1
                    logger.info(f"[冷却中] 跳过检测周期，剩余 {self.skip_cycles} 周期")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 查找微信窗口
                window = self.ocr.find_wechat_window()
                if not window:
                    logger.warning("[等待] 未找到微信窗口...")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 截取聊天区域
                screenshot = self.ocr.capture_chat_area(window)
                if not screenshot:
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 检测变化区域
                bbox = self.ocr.get_change_bbox(
                    screenshot,
                    self.config['CHANGE_DETECTION_MIN_AREA']
                )

                if not bbox:
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 裁剪变化区域并扩展边界
                expanded_bbox = self.ocr.expand_bbox(
                    bbox,
                    screenshot.size,
                    self.config['BBOX_MARGIN']
                )
                region = screenshot.crop(expanded_bbox)

                logger.info(f"\n[{datetime.now().strftime('%H:%M:%S')}] 检测到新消息，识别中...")

                # OCR识别
                text = self.ocr.ocr_recognize(
                    region,
                    self.config['FILTER_GREEN_BUBBLES']
                )

                # 提取最新消息
                message = self.ocr.extract_latest_message(text)

                if not message:
                    logger.warning("[跳过] 未识别到有效消息")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                logger.info(f"[收到] {message[:80]}{'...' if len(message) > 80 else ''}")

                # 去重检查
                if self.is_duplicate_message(message):
                    logger.info("[跳过] 重复消息")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 检查是否为自己的回复
                if self.is_recent_self_reply(message):
                    logger.info("[跳过] 检测到可能是机器人自身回复")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 检查是否应该回复
                if not self.should_auto_reply():
                    logger.info("[跳过] 当前时间段不自动回复")
                    time.sleep(self.config['CHECK_INTERVAL'])
                    continue

                # 生成AI回复
                logger.info("[AI思考] 生成回复中...")
                context = '\n'.join(self.conversation_context[-6:]) if self.conversation_context else None
                reply = self.ai.generate_reply(message, context)

                logger.info(f"[回复] {reply}")

                # 发送消息
                time.sleep(1)  # 模拟真人延迟

                if WeChatUI.send_message(window, reply):
                    logger.info("[成功] 消息已发送")
                    self.update_context(message, reply)

                    # 设置跳过周期
                    self.skip_cycles = self.config['SKIP_CYCLES_AFTER_SEND']
                else:
                    logger.error("[失败] 消息发送失败")

                time.sleep(self.config['CHECK_INTERVAL'])

            except KeyboardInterrupt:
                logger.info("\n\n机器人已停止运行")
                break

            except Exception as e:
                logger.error(f"\n[异常] {e}", exc_info=self.config['DEBUG_MODE'])
                time.sleep(self.config['CHECK_INTERVAL'])

# ==================== 主程序入口 ====================
def main():
    """主程序入口"""
    # 从环境变量读取API Key
    api_key = os.getenv('DEEPSEEK_API_KEY') or CONFIG['API_KEY']

    # 验证API Key
    if not api_key or api_key == 'YOUR_DEEPSEEK_API_KEY':
        logger.error("错误：请配置DeepSeek API Key")
        logger.info("方式1: 修改脚本中的 CONFIG['API_KEY']")
        logger.info("方式2: 设置环境变量 DEEPSEEK_API_KEY")
        return

    CONFIG['API_KEY'] = api_key

    # 创建并运行机器人
    bot = WeChatAIBot(CONFIG)
    bot.run()

if __name__ == "__main__":
    main()

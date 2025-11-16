# 微信OCR自动回复机器人 - DeepSeek AI版

基于OCR识别 + DeepSeek AI的智能微信自动回复工具，代码仅247行。

## 特性

✅ **AI智能回复** - 使用DeepSeek大模型生成自然、上下文相关的回复  
✅ **OCR识别** - 无需Hook微信，安全性高  
✅ **消息去重** - 避免重复回复同一条消息  
✅ **上下文记忆** - 支持多轮对话上下文  
✅ **工作时间控制** - 可设置自动回复的时间段  
✅ **代码精简** - 250行以内，易于理解和修改  

## 环境要求

- Python 3.7+
- Windows 10/11（支持微信PC版）
- Tesseract OCR
- DeepSeek API Key

## 安装步骤

### 1. 安装Python依赖

```bash
pip install pillow pytesseract pyautogui pygetwindow openai pyperclip
```

### 2. 安装Tesseract OCR

**Windows:**
1. 下载安装包：https://github.com/UB-Mannheim/tesseract/wiki
2. 安装到默认路径：`C:\Program Files\Tesseract-OCR`
3. 下载中文语言包：
   - 访问：https://github.com/tesserac
   - t-ocr/tessdata
   - 下载 `chi_sim.traineddata`
   - 放到：`C:\Program Files\Tesseract-OCR\tessdata`

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim
```

### 3. 获取DeepSeek API Key

1. 访问：https://platform.deepseek.com/
2. 注册并登录
3. 进入"API Keys"页面
4. 创建新的API Key
5. 复制保存（只显示一次）

## 配置说明

### 方式1：修改脚本配置

编辑 `wechat_ai_bot.py`，修改配置区：

```python
CONFIG = {
    'API_KEY': 'sk-xxxxxxxxxxxxxxxx',  # 你的DeepSeek API Key
    'TESSERACT_PATH': r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    'CHECK_INTERVAL': 3,  # 检测间隔（秒）
}
```

### 方式2：使用环境变量（推荐）

**Windows:**
```cmd
set DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
python wechat_ai_bot.py
```

**macOS/Linux:**
```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
python wechat_ai_bot.py
```

## 使用方法

### 1. 启动微信并打开聊天窗口

1. 登录微信PC版
2. 打开需要自动回复的**单个聊天窗口**
3. 保持聊天窗口可见（不要最小化）

### 2. 运行脚本

```bash
python wechat_ai_bot.py
```

### 3. 观察运行状态

```
==================================================
微信AI自动回复机器人启动
AI模型: deepseek-chat
检测间隔: 3秒
==================================================

[15:30:45] 检测到新消息，识别中...
[收到] 你好，在吗？
[AI思考] 生成回复中...
[回复] 你好！我在的，有什么可以帮助你的吗？
[成功] 消息已发送
```

### 4. 停止运行

按 `Ctrl + C` 停止脚本

## 高级配置

### 1. 工作时间限制

只在非工作时间自动回复：

```python
CONFIG = {
    'WORK_HOURS': {'start': 9, 'end': 18},  # 9:00-18:00 不自动回复
}
```

### 2. 自定义AI提示词

修改 `DeepSeekAI` 类中的 `system_prompt`：

```python
self.system_prompt = """你是我的私人助理，帮我处理微信消息。
回复风格：专业、简洁、友好
特殊规则：
- 遇到紧急词汇（如"紧急"、"ASAP"）立即提醒我
- 会议邀请自动确认时间
- 其他消息简要回复"""
```

### 3. 调整OCR识别区域

根据你的屏幕分辨率和微信窗口大小，可能需要调整识别区域：

```python
# 在 WeChatOCR.capture_chat_area() 方法中调整
x = window.left + int(window.width * 0.35)  # 左边距比例
y = window.top + 80                          # 上边距像素
width = int(window.width * 0.62)            # 宽度比例
height = int(window.height * 0.55)          # 高度比例
```

### 4. 提高OCR准确率

如果识别不准确，可以：

1. **提高图像对比度**（在代码中已包含灰度转换）
2. **使用PaddleOCR替代**（准确率更高，但需要额外依赖）
3. **调整Tesseract配置**

## 常见问题

### Q1: 提示"未找到微信窗口"

**解决方案：**
- 确保微信已登录
- 微信窗口标题是"微信"（不要改名）
- Windows语言设置为中文

### Q2: OCR识别不准确

**解决方案：**
1. 确认中文语言包已安装
2. 调整聊天窗口大小（推荐全屏或较大窗口）
3. 使用默认微信主题（浅色背景）
4. 增大 `CHECK_INTERVAL` 避免识别到半条消息

### Q3: 消息发送失败

**解决方案：**
- 确认聊天窗口在最前面
- 调整点击坐标（根据你的屏幕分辨率）
- 确保输入法是中文模式

### Q4: DeepSeek API调用失败

**解决方案：**
```python
# 检查错误信息
[AI错误] Error code: 401 - 无效的API Key
[AI错误] Error code: 429 - 超过速率限制
[AI错误] Error code: 500 - 服务器错误
```

- 401: 检查API Key是否正确
- 429: 降低请求频率或升级账户
- 500: 稍后重试

### Q5: 如何避免封号？

**安全建议：**
1. ✅ 使用小号测试
2. ✅ 设置合理的回复延迟（1-3秒）
3. ✅ 避免高频回复（间隔3秒以上）
4. ✅ 不要在群聊中使用
5. ⚠️ 仅用于个人便利，不要商业用途

## 成本估算

DeepSeek API定价（2024年数据）：
- 输入：¥1/百万tokens
- 输出：¥2/百万tokens

**示例计算：**
- 每条消息约消耗 200 tokens
- 100条消息 ≈ ¥0.04
- 1000条消息 ≈ ¥0.40

**非常便宜！** 日常使用几乎可以忽略不计。

## 代码结构

```
wechat_ai_bot.py (247行)
├── CONFIG              # 配置字典
├── DeepSeekAI          # AI回复生成
├── WeChatOCR           # OCR消息识别
├── WeChatUI            # UI自动化操作
└── WeChatAIBot         # 主控制逻辑
```

## 扩展建议

### 1. 添加更多AI能力

```python
# 集成文件下载、图片识别等
def generate_reply(self, message, context=None, image=None):
    if image:
        # 使用DeepSeek视觉模型识别图片
        pass
```

### 2. 多聊天窗口支持

```python
# 通过窗口标题或位置区分不同聊天
def get_active_chat_name(self, window):
    # OCR识别顶部聊天对象名称
    pass
```

### 3. 消息持久化

```python
import json

def save_conversation(self, filename='chat_history.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(self.conversation_context, f, ensure_ascii=False)
```

## 免责声明

⚠️ 本项目仅供学习和个人使用  
⚠️ 使用第三方工具操作微信可能违反服务条款  
⚠️ 请遵守相关法律法规，不要用于非法用途  
⚠️ 因使用本工具造成的任何后果，作者概不负责  

## 技术支持

遇到问题？
1. 检查本文档的"常见问题"部分
2. 确认所有依赖已正确安装
3. 查看控制台错误信息
4. 调整配置参数重试

## License

MIT License - 自由使用、修改、分发

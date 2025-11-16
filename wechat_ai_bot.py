#!/usr/bin/env python3
import os,re,time,hashlib;from datetime import datetime;from difflib import SequenceMatcher
import numpy as np,pyautogui,pygetwindow as gw,pytesseract,pyperclip
from PIL import Image,ImageChops,ImageDraw,ImageFilter,ImageOps;from openai import OpenAI

# 紧凑配置
CONFIG={'API_KEY':'YOUR_API_KEY','API_BASE':'https://api.deepseek.com','MODEL':'deepseek-chat','TESSERACT_PATH':r'C:\Program Files\Tesseract-OCR\tesseract.exe','OCR_LANG':'chi_sim','CHECK_INTERVAL':3,'AUTO_REPLY_ENABLED':True,'WORK_HOURS':None,'REPLY_COOLDOWN':10,'SKIP_CYCLES_AFTER_SEND':3,'FILTER_GREEN_BUBBLES':True}

class DeepSeekAI:
    def __init__(self,api_key,base_url,model): self.client=OpenAI(api_key=api_key,base_url=base_url); self.model=model; self.system_prompt="你是微信自动回复助手，50字内友好简洁地回复，问候友好回应，问题简要回答或稍后处理，保持礼貌专业。"
    def generate_reply(self,message,context=None):
        try:
            msgs=[{"role":"system","content":self.system_prompt}]
            if context: msgs.append({"role":"user","content":f"对话历史：{context}"})
            msgs.append({"role":"user","content":f"收到消息：{message}\n请生成回复："})
            r=self.client.chat.completions.create(model=self.model,messages=msgs,max_tokens=100,temperature=0.7)
            return r.choices[0].message.content.strip()
        except Exception as e: print(f"[AI错误] {e}"); return "收到你的消息，稍后回复~"

class WeChatOCR:
    def __init__(self,tesseract_path,lang): 
        if tesseract_path: pytesseract.pytesseract.tesseract_cmd=tesseract_path
        self.lang=lang; self.last_screenshot=None
    def find_wechat_window(self): ws=gw.getWindowsWithTitle('微信'); return ws[0] if ws else None
    def capture_chat_area(self,window):
        try:
            window.activate(); time.sleep(0.1); x=window.left+int(window.width*0.35); y=window.top+80; w=int(window.width*0.62); h=int(window.height*0.70)
            return pyautogui.screenshot(region=(x,y,w,h))
        except Exception as e: print(f"[截图错误] {e}"); return None
    def get_change_bbox(self,current_screenshot,min_area=600):
        if self.last_screenshot is None: self.last_screenshot=current_screenshot; return None
        try:
            diff=ImageChops.difference(current_screenshot,self.last_screenshot); bbox=diff.getbbox(); self.last_screenshot=current_screenshot
            if not bbox: return None
            return bbox if (bbox[2]-bbox[0])*(bbox[3]-bbox[1])>=min_area else None
        except Exception: self.last_screenshot=current_screenshot; return None
    @staticmethod
    def expand_bbox(bbox,image_size,margin=20):
        x0,y0,x1,y1=bbox; w,h=image_size; return (max(0,x0-margin),max(0,y0-margin),min(w,x1+margin),min(h,y1+margin))
    def mask_green_bubbles(self,image):
        try:
            arr=np.array(image); r,g,b=arr[:,:,0],arr[:,:,1],arr[:,:,2]; green=(g>170)&(r<190)&((g-r)>40)
            try:
                from scipy.ndimage import binary_dilation; green=binary_dilation(green,iterations=20)
            except ImportError: print("[提示] pip install scipy 可提升识别准确率")
            arr[green]=[245,245,245]; return Image.fromarray(arr)
        except Exception as e: print(f"[屏蔽绿色错误] {e}"); return image
    def keep_text_regions(self,image):
        try:
            inv=ImageOps.invert(image.convert('L')); blur=inv.filter(ImageFilter.BoxBlur(1)); arr=np.array(blur); th=np.percentile(arr,70)
            bw=blur.point(lambda p:255 if p>th else 0); dil=bw.filter(ImageFilter.MaxFilter(9)); bg=Image.new('RGB',image.size,(245,245,245))
            return Image.composite(image,bg,dil)
        except Exception as e: print(f"[文本掩膜错误] {e}"); return image
    def ocr_recognize(self,image,filter_green=True):
        try:
            if filter_green: image=self.mask_green_bubbles(image)
            image=self.keep_text_regions(image); w,h=image.size; image=image.resize((max(1,int(w*1.7)),max(1,int(h*1.7))),Image.BILINEAR)
            gray=ImageOps.autocontrast(image.convert('L')); cfg=r'--oem 3 --psm 4'
            return pytesseract.image_to_string(gray,lang='chi_sim',config=cfg).strip()
        except Exception as e: print(f"[OCR错误] {e}"); return ""
    def extract_latest_message(self,full_text):
        if not full_text: return None
        ls=[l.strip() for l in full_text.split('\n') if l.strip()]
        if not ls: return None
        noise=['复制','转发','撤回','编辑','收藏','消息','表情','图片','视频','文件','链接','语音','已读','未读']
        fl=[l for l in ls if not re.match(r'^\d{1,2}:\d{2}$',l) and len(l)>=2 and not any(k in l for k in noise)]
        return ' '.join(fl[-2:]) if fl else None

class WeChatUI:
    @staticmethod
    def send_message(window,text):
        try:
            x=window.left+int(window.width*0.65); y=window.top+window.height-120; pyautogui.click(x,y); time.sleep(0.15); pyautogui.hotkey('ctrl','a'); time.sleep(0.05)
            pyperclip.copy(text); pyautogui.hotkey('ctrl','v'); time.sleep(0.2); pyautogui.press('enter'); return True
        except Exception as e: print(f"[发送错误] {e}"); return False

class WeChatAIBot:
    def __init__(self,config):
        self.config=config; self.ai=DeepSeekAI(config['API_KEY'],config['API_BASE'],config['MODEL']); self.ocr=WeChatOCR(config['TESSERACT_PATH'],config['OCR_LANG'])
        self.message_history=set(); self.conversation_context=[]; self.last_reply_time=0; self.last_reply_content=""; self.skip_cycles=0
    def get_message_hash(self,message): return hashlib.md5(message.encode()).hexdigest()
    def is_duplicate_message(self,message):
        h=self.get_message_hash(message)
        if h in self.message_history: return True
        self.message_history.add(h)
        if len(self.message_history)>50: self.message_history.pop()
        return False
    def is_recent_self_reply(self,message):
        now=time.time(); cd=self.config.get('REPLY_COOLDOWN',10)
        if now-self.last_reply_time<cd:
            if message.strip()==self.last_reply_content.strip(): return True
            if any(k in message for k in ['回复','稍后','收到','系统']): return True
            if SequenceMatcher(None,message.strip(),self.last_reply_content.strip()).ratio()>0.85: return True
        return False
    def should_auto_reply(self):
        if not self.config['AUTO_REPLY_ENABLED']: return False
        wh=self.config.get('WORK_HOURS'); 
        if wh and wh['start']<=datetime.now().hour<wh['end']: return False
        return True
    def update_context(self,message,reply):
        self.conversation_context.append(f"用户: {message}"); self.conversation_context.append(f"我: {reply}")
        if len(self.conversation_context)>12: self.conversation_context=self.conversation_context[-12:]
        self.last_reply_time=time.time(); self.last_reply_content=reply
    def run(self):
        print("="*36+f"\n机器人启动 | 模型:{self.config['MODEL']} | 间隔:{self.config['CHECK_INTERVAL']}s\n"+"="*36)
        print("请确保：微信登录且聊天窗口可见；API Key 已配置。\nCtrl+C 结束。\n")
        while True:
            try:
                if self.skip_cycles>0: self.skip_cycles-=1; print(f"[冷却] 跳过，剩余 {self.skip_cycles}"); time.sleep(self.config['CHECK_INTERVAL']); continue
                window=self.ocr.find_wechat_window()
                if not window: print("[等待] 未找到微信窗口"); time.sleep(self.config['CHECK_INTERVAL']); continue
                shot=self.ocr.capture_chat_area(window)
                if not shot: time.sleep(self.config['CHECK_INTERVAL']); continue
                bbox=self.ocr.get_change_bbox(shot)
                if not bbox: time.sleep(self.config['CHECK_INTERVAL']); continue
                region=shot.crop(self.ocr.expand_bbox(bbox,shot.size,20))
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 识别中..")
                text=self.ocr.ocr_recognize(region,self.config.get('FILTER_GREEN_BUBBLES',True))
                msg=self.ocr.extract_latest_message(text)
                if not msg: print("[跳过] 无有效消息"); time.sleep(self.config['CHECK_INTERVAL']); continue
                print(f"[收到] {msg[:80]}...")
                if self.is_duplicate_message(msg): print("[跳过] 重复消息"); time.sleep(self.config['CHECK_INTERVAL']); continue
                if self.is_recent_self_reply(msg): print("[跳过] 可能是机器人自身回复"); time.sleep(self.config['CHECK_INTERVAL']); continue
                if not self.should_auto_reply(): print("[跳过] 非自动回复时段"); time.sleep(self.config['CHECK_INTERVAL']); continue
                print("[AI] 生成回复.."); ctx='\n'.join(self.conversation_context[-6:]) if self.conversation_context else None
                reply=self.ai.generate_reply(msg,ctx); print(f"[回复] {reply}"); time.sleep(1)
                if WeChatUI.send_message(window,reply): print("[成功] 已发送"); self.update_context(msg,reply); self.skip_cycles=self.config.get('SKIP_CYCLES_AFTER_SEND',3)
                else: print("[失败] 发送失败")
                time.sleep(self.config['CHECK_INTERVAL'])
            except KeyboardInterrupt: print("\n已停止"); break
            except Exception as e: print(f"\n[异常] {e}"); time.sleep(self.config['CHECK_INTERVAL'])

if __name__=="__main__":
    api=os.getenv('DEEPSEEK_API_KEY') or CONFIG['API_KEY']
    if not api or api=='YOUR_DEEPSEEK_API_KEY': print("错误：请配置 DeepSeek API Key（环境变量 DEEPSEEK_API_KEY 或修改脚本 CONFIG['API_KEY']）"); raise SystemExit(1)
    CONFIG['API_KEY']=api; WeChatAIBot(CONFIG).run()

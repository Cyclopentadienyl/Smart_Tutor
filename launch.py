import sys
import os
import traceback

# --- 關鍵修正：強制將工作目錄設定為腳本所在目錄 ---
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 將專案根目錄加入 Python 路徑，確保能找到 src 模組
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # 將所有 import 也包在 try 區塊內，防止因為套件缺失導致閃退
    from src.ui_layout import create_ui

    if __name__ == "__main__":
        print("--- 啟動 AI 家教進度分派系統 ---")
        print("正在初始化介面...")
        
        # 建立 Gradio App
        demo = create_ui()
        
        # 啟動 Web Server
        # server_name="0.0.0.0" 允許區網訪問
        # inbrowser=True 自動打開瀏覽器
        print("啟動成功！網頁介面即將開啟...")
        demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)

except ImportError as e:
    print("\n❌ [啟動失敗] 找不到必要的模組/套件！")
    print("可能原因：未正確安裝依賴，或虛擬環境未啟動。")
    print(f"錯誤細節：{e}")
    traceback.print_exc()
    input("\n🔴 請按 Enter 鍵退出視窗...")

except Exception as e:
    print("\n❌ [系統錯誤] 發生未預期的錯誤：")
    traceback.print_exc()
    input("\n🔴 請按 Enter 鍵退出視窗...")
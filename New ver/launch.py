import os
import sys
import logging

# ==========================================
# 💉 最終極熱修復 (The Lobotomy Patch)
# ==========================================
# 問題：Gradio Client 在舊環境下解析 Schema 會崩潰 (APIInfoParseError)
# 解法：我們直接 "閹割" 掉它的解析功能。
#       我們定義一個永遠回傳 "Any" 的函數，並強行覆蓋掉原本的邏輯。
#       這對網頁 UI (Browser) 完全沒影響，只會影響 API 文件 (我們不需要)。
# ==========================================
try:
    import gradio_client.utils
    
    print("🔧 正在執行深度熱修復 (Deep Patch)...")
    
    # 定義一個什麼都不做，只回傳字串的 "啞巴函數"
    def dummy_schema_parser(*args, **kwargs):
        return "Any"  # 無論原本要算什麼，直接回傳 "Any" 騙過系統

    # 強制覆蓋兩個關鍵解析函式
    gradio_client.utils._json_schema_to_python_type = dummy_schema_parser
    gradio_client.utils.json_schema_to_python_type = dummy_schema_parser
    
    print("✅ 修復完成：API Schema 解析器已被停用。")
except Exception as e:
    print(f"⚠️ 修復補丁警告: {e}")
# ==========================================


# --- 設定環境變數 ---
# 強制忽略所有 Proxy 設定，確保 Python 不會嘗試連線到公司 Proxy
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'

# --- 設定路徑 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from src.ui_layout import create_ui
except ImportError as e:
    print(f"❌ 模組匯入失敗: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("🔄 初始化系統中...")
    
    # 關閉大部分的 log，避免干擾
    logging.getLogger("gradio").setLevel(logging.WARNING)
    
    demo = create_ui()
    
    print("🚀 伺服器啟動中 (全開放模式)...")
    print("---------------------------------------------------------")
    print("ℹ️  請在瀏覽器輸入以下網址進行連線：")
    print("👉  http://localhost:7860")
    print("---------------------------------------------------------")

    try:
        demo.launch(
            inbrowser=True,       
            server_name="0.0.0.0", # 使用 0.0.0.0 增加綁定成功率
            server_port=7860,
            share=False,          # 關閉分享 (因為 Proxy 擋住了)
            show_api=False,       # 再次宣告關閉 API
            show_error=True
        )
    except Exception as e:
        print(f"❌ 啟動發生未預期錯誤: {e}")
        input("請按 Enter 鍵關閉視窗...")
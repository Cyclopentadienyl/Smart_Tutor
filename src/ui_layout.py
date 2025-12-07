import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

# 引用原有的模組
from src.ai_engine import TutorAI
from src.data_manager import DataManager
from src import config

# 引用新加入的排程模組 (請確保 src/scheduler.py 已建立)
try:
    from src.scheduler import CurriculumScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("⚠️ Warning: scheduler.py not found. Schedule features will be disabled.")

# 設定中文字型 (保留原版設定)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

# 初始化各個管理器
dm = DataManager()
ai = TutorAI()
scheduler = CurriculumScheduler() if SCHEDULER_AVAILABLE else None


def create_ui():
    with gr.Blocks(title="AI Smart Tutor System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎓 AI Smart Tutor: 智慧家教與專案管理系統")

        # 用於跨 Tab 共享資料的 State
        # 雖然 DataManager 會存檔，但用 State 可以讓戰情室與資料區同步
        shared_df = gr.State(pd.DataFrame())

        with gr.Tabs():
            
            # =========================================
            # Tab 1: 🆕 全班戰情室 (New Feature)
            # =========================================
            with gr.Tab("🏫 全班戰情室 (Class Monitor)"):
                gr.Markdown("### 📊 學生狀態總覽與排程干預")
                with gr.Row():
                    # 左側：學生紅綠燈列表
                    with gr.Column(scale=4):
                        btn_refresh_dashboard = gr.Button("🔄 掃描全班狀態 (Scan)", variant="primary")
                        student_table = gr.Dataframe(
                            headers=["ID", "Name", "Risk", "Current Topic", "Accuracy"],
                            datatype=["str", "str", "str", "str", "number"],
                            interactive=False,
                            label="點擊任一學生查看詳情 👇"
                        )
                    
                    # 右側：甘特圖與介入
                    with gr.Column(scale=6):
                        gr.Markdown("### 📅 AI 學習路徑規劃")
                        selected_student_id = gr.Textbox(label="目前選取學生", interactive=False, value="未選取")
                        gantt_chart = gr.Plot(label="動態甘特圖")
                        
                        with gr.Group():
                            ai_rationale = gr.Textbox(label="🤖 AI 排程邏輯 (解釋性)", lines=2, interactive=False)
                            btn_optimize = gr.Button("✨ 一鍵優化此路徑 (Re-schedule)")

            # =========================================
            # Tab 2: 📊 資料匯入與視覺化 (Original Tab 1)
            # =========================================
            with gr.Tab("📊 資料庫管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 1. 資料來源")
                        btn_load = gr.Button("🔄 重置為模擬資料", variant="secondary")

                        gr.Markdown("### 2. 外部匯入")
                        file_input = gr.File(label="上傳資料表", file_count="single")
                        btn_analyze_file = gr.Button("📂 分析上傳檔", variant="primary")
                        import_log = gr.Textbox(label="狀態", interactive=False)

                    with gr.Column(scale=3):
                        data_display = gr.Dataframe(interactive=False, label="資料預覽")

                with gr.Row():
                    plot_scatter = gr.Plot(label="學習狀態分佈 (含回歸分析)")
                    plot_bar = gr.Plot(label="推薦統計")

            # =========================================
            # Tab 3: 🤖 AI 分析核心 (Original Tab 2)
            # =========================================
            with gr.Tab("🧠 AI 核心訓練"):
                gr.Markdown("### 執行 AI 演算法")
                btn_run_ai = gr.Button("🚀 開始 AI 訓練與分析", variant="primary")
                result_log = gr.Textbox(label="日誌", interactive=False)

            # =========================================
            # Tab 4: 🧑‍🎓 個人能力診斷 (Original Tab 3)
            # =========================================
            with gr.Tab("🧑‍🎓 個人能力診斷"):
                gr.Markdown("### 輸入多維度數據")
                with gr.Row():
                    with gr.Column(scale=1):
                        in_accuracy = gr.Slider(0.0, 1.0, value=0.75, label="正確率 (Accuracy)", step=0.01)
                        in_time = gr.Slider(5.0, 60.0, value=15.0, label="平均完成時間", step=0.5)
                        in_pace = gr.Slider(0.0, 30.0, value=10.0, label="學習步調", step=0.5)
                        in_attend = gr.Slider(0.0, 1.0, value=0.9, label="出席率", step=0.01)
                        in_hw = gr.Slider(0.0, 1.0, value=0.8, label="作業完成率", step=0.01)

                        gr.Markdown("#### 錯誤類型")
                        with gr.Row():
                            in_err_read = gr.Number(value=2, label="Reading", precision=0)
                            in_err_vocab = gr.Number(value=1, label="Vocab", precision=0)
                            in_err_logic = gr.Number(value=3, label="Logic", precision=0)
                        in_mean_score = gr.Number(value=80, label="平均分數", precision=0)

                        btn_predict_user = gr.Button("🔮 開始診斷", variant="primary")

                    with gr.Column(scale=1):
                        output_result = gr.Markdown("### ⏳ 等待輸入...")
                        output_advice = gr.Markdown("")

        # =========================================
        # Logic Functions (邏輯實作區)
        # =========================================

        # --- 1. 原版繪圖函數 (完全保留) ---
        def draw_plots(df: pd.DataFrame):
            # 防呆：檢查必要欄位
            cols_to_check = [col for col in [config.COL_ACCURACY, config.COL_AVG_TIME] if col in df.columns]
            df = df.dropna(subset=cols_to_check)

            # Scatter Plot (Matplotlib)
            fig_scatter = None
            if not df.empty:
                try:
                    plt.close("all")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    x_col = config.COL_AVG_TIME
                    y_col = config.COL_ACCURACY
                    group_col = config.COL_GROUP if config.COL_GROUP in df.columns else None

                    if group_col:
                        groups = sorted(df[group_col].unique())
                        colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
                        for idx, group in enumerate(groups):
                            sub_df = df[df[group_col] == group]
                            if len(sub_df) == 0: continue
                            ax.scatter(sub_df[x_col], sub_df[y_col], label=f"Group {group}", color=colors[idx], alpha=0.7, s=80)
                            # 回歸線
                            if len(sub_df) > 1:
                                try:
                                    slope, intercept = np.polyfit(sub_df[x_col], sub_df[y_col], 1)
                                    x_line = np.linspace(sub_df[x_col].min(), sub_df[x_col].max(), 100)
                                    ax.plot(x_line, slope * x_line + intercept, color=colors[idx], linestyle="--", alpha=0.9)
                                except: pass
                        ax.legend(title="Groups")
                    else:
                        ax.scatter(df[x_col], df[y_col], c="tab:blue", alpha=0.7, s=80)
                    
                    ax.set_xlabel("平均完成時間 (分)")
                    ax.set_ylabel("正確率 (Accuracy)")
                    ax.set_title("學生學習狀態分佈", fontname="Microsoft JhengHei", fontsize=14)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    fig.tight_layout()
                    fig_scatter = fig
                except Exception as exc:
                    print(f"Plot Error: {exc}")

            # Bar Plot (Plotly)
            fig_bar = None
            if not df.empty and config.COL_RECOMMENDED_LEVEL in df.columns:
                fig_bar = px.histogram(
                    df,
                    x=config.COL_RECOMMENDED_LEVEL,
                    title="AI 推薦難度分佈",
                    text_auto=True,
                    color=config.COL_RECOMMENDED_LEVEL,
                )
            
            return fig_scatter, fig_bar

        # --- 2. 原版資料載入 (完全保留) ---
        def load_default_data():
            df = dm.load_data()
            fig_s, fig_b = draw_plots(df)
            return df, df, fig_s, fig_b, "已重置為模擬資料" # 更新 shared_df

        def process_uploaded_file(file_obj):
            if file_obj is None: return None, None, None, None, "請先選擇檔案"
            file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
            df = dm.load_uploaded_file(file_path)
            if df.empty or config.COL_ACCURACY not in df.columns:
                return None, None, None, None, "檔案格式錯誤"
            
            df, log = ai.run_inference_only(df)
            fig_s, fig_b = draw_plots(df)
            return df, df, fig_s, fig_b, log # 更新 shared_df

        # --- 3. 原版訓練流程 (完全保留) ---
        def run_training_pipeline():
            df = dm.load_data()
            processed_df, msg = ai.run_analysis_pipeline(df)
            dm.save_results(processed_df)
            fig_s, fig_b = draw_plots(processed_df)
            return processed_df, processed_df, msg, fig_s, fig_b # 更新 shared_df

        # --- 4. 原版單人預測 (完全保留) ---
        def predict_user(acc, time, pace, att, hw, e_read, e_vocab, e_logic, m_score):
            result = str(ai.predict_single(acc, time, pace, att, hw, e_read, e_vocab, e_logic, m_score))
            advice = ""
            if "Hard" in result: advice = "💡 表現優異，建議挑戰進階課程。"
            elif "Easy" in result: advice = "💡 建議加強基礎練習。"
            else: advice = "💡 進度穩定，適合標準課程。"
            return f"# 🎯 結果：{result}", advice

        # --- 5. 🆕 新增：戰情室邏輯 ---
        def refresh_dashboard():
            """掃描資料，計算風險，更新列表"""
            df = dm.load_data()
            # 確保有 AI 預測結果
            df, _ = ai.run_inference_only(df)
            
            # 呼叫 AI 的風險評估 (需確保 ai_engine.py 有 batch_evaluate_risk 方法)
            # 若無，則使用簡易 fallback
            if hasattr(ai, "batch_evaluate_risk"):
                df = ai.batch_evaluate_risk(df)
            else:
                # Fallback logic just in case user didn't update ai_engine.py
                df[config.COL_RISK_LEVEL] = df[config.COL_ACCURACY].apply(lambda x: "🔴 High Risk" if x < 0.6 else "🟢 On Track")
                df["Current Topic"] = "一般課程"
            
            # 準備顯示用的表格 (只取關鍵欄位)
            # 檢查欄位是否存在，避免 KeyError
            cols = [config.COL_STUDENT_ID, config.COL_NAME, config.COL_RISK_LEVEL, "Current_Topic", config.COL_ACCURACY]
            valid_cols = [c for c in cols if c in df.columns]
            
            display_df = df[valid_cols]
            return df, display_df

        def on_student_select(evt: gr.SelectData, full_df):
            """點擊學生後，生成排程圖"""
            if not SCHEDULER_AVAILABLE:
                return "Scheduler module missing", None, "請檢查 src/scheduler.py"
            
            if full_df is None or full_df.empty:
                return "無資料", None, "請先執行掃描"
            
            # 根據點擊的 Row Index 找到學生
            try:
                row_index = evt.index[0]
                student_row = full_df.iloc[row_index]
                s_id = student_row[config.COL_STUDENT_ID]
                s_name = student_row[config.COL_NAME]
                
                # 生成甘特圖
                df_sched, msg = scheduler.generate_student_schedule(student_row)
                fig = scheduler.plot_gantt(df_sched)
                
                # 生成解釋文字
                risk = student_row.get(config.COL_RISK_LEVEL, "未知")
                pace = student_row.get(config.COL_LEARNING_PACE, 10)
                reason = f"【{risk}】 學生步調係數: {pace}。\n系統已依據其弱點重新權衡課程工時 (Resource Leveling)。"
                
                return f"{s_id} - {s_name}", fig, reason
            except Exception as e:
                return "Error", None, str(e)

        def optimize_schedule_action(s_id_str, full_df):
            """模擬老師介入優化"""
            if not SCHEDULER_AVAILABLE: return None, "模組缺失"
            if not s_id_str or "未選取" in s_id_str: return None, "請先選擇學生"
            
            try:
                s_id = s_id_str.split(" - ")[0]
                # 找到該學生資料
                student_row = full_df[full_df[config.COL_STUDENT_ID] == s_id].iloc[0].copy()
                
                # 模擬：強制調整參數
                student_row[config.COL_LEARNING_PACE] = float(student_row[config.COL_LEARNING_PACE]) * 1.5
                
                df_sched, msg = scheduler.generate_student_schedule(student_row)
                fig = scheduler.plot_gantt(df_sched)
                return fig, f"✅ 已強制調整該生權重 (Pace x 1.5) 並重新排程。\n{msg}"
            except Exception as e:
                return None, str(e)

        # =========================================
        # Event Bindings (事件綁定)
        # =========================================
        
        # Tab 1: 戰情室事件
        btn_refresh_dashboard.click(refresh_dashboard, outputs=[shared_df, student_table])
        student_table.select(on_student_select, inputs=[shared_df], outputs=[selected_student_id, gantt_chart, ai_rationale])
        btn_optimize.click(optimize_schedule_action, inputs=[selected_student_id, shared_df], outputs=[gantt_chart, ai_rationale])

        # Tab 2: 資料管理事件 (注意：這裡同時更新 shared_df)
        btn_load.click(load_default_data, outputs=[shared_df, data_display, plot_scatter, plot_bar, import_log])
        btn_analyze_file.click(process_uploaded_file, inputs=[file_input], outputs=[shared_df, data_display, plot_scatter, plot_bar, import_log])

        # Tab 3: 訓練事件
        btn_run_ai.click(run_training_pipeline, outputs=[shared_df, data_display, result_log, plot_scatter, plot_bar])

        # Tab 4: 預測事件
        btn_predict_user.click(
            fn=predict_user,
            inputs=[in_accuracy, in_time, in_pace, in_attend, in_hw, in_err_read, in_err_vocab, in_err_logic, in_mean_score],
            outputs=[output_result, output_advice]
        )

        # 初始載入
        app.load(fn=load_default_data, outputs=[shared_df, data_display, plot_scatter, plot_bar, import_log])

    return app
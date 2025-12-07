import gradio as gr
import plotly.express as px
from src.data_manager import DataManager
from src.ai_engine import TutorAI
from src import config

# 初始化後端模組
dm = DataManager()
ai = TutorAI()

def create_ui():
    """
    建立 Gradio 介面
    """
    display_columns = [
        config.COL_STUDENT_ID,
        config.COL_AVG_SCORE,
        config.COL_AVG_TIME,
        config.COL_PROGRESS,
        config.COL_GROUP,
        config.COL_PROFICIENCY_SCORE,
        config.COL_RECOMMENDED_LEVEL,
        config.COL_WEAKNESS,
        config.COL_PERSONALITY,
    ]

    with gr.Blocks(title="AI 家教進度分派系統") as app:
        gr.Markdown("# 🎓 AI 輔助家教進度分派系統 (Phase 1)")
        
        with gr.Tabs():
            # --- Tab 1: 資料總覽 ---
            with gr.TabItem("📊 學生資料與視覺化"):
                with gr.Row():
                    btn_load = gr.Button("🔄 讀取/重置 資料", variant="secondary")
                    btn_save = gr.Button("💾 儲存結果", interactive=False) # 初始不可按
                
                data_display = gr.Dataframe(
                    headers=display_columns,
                    datatype=[
                        "str",
                        "number",
                        "number",
                        "number",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                    ],
                    interactive=False,
                )
                
                with gr.Row():
                    plot_scatter = gr.Plot(label="進度分群圖 (成績 vs 時間)")
                    plot_bar = gr.Plot(label="各組弱點分佈")

            # --- Tab 2: AI 分析核心 ---
            with gr.TabItem("🤖 AI 分派與運算"):
                gr.Markdown("### 執行 AI 演算法進行分群與推薦")
                btn_run_ai = gr.Button("🚀 開始 AI 分析 (Train & Predict)", variant="primary")
                result_log = gr.Textbox(label="系統日誌", interactive=False)

        # --- 事件處理 (Event Handling) ---
        
        def _format_for_table(df):
            missing_columns = [col for col in display_columns if col not in df.columns]
            for col in missing_columns:
                df[col] = None
            return df.reindex(columns=display_columns)

        def load_and_plot():
            """讀取資料並畫圖"""
            df = dm.load_data()
            
            # 畫散佈圖
            fig_scatter = px.scatter(
                df, 
                x=config.COL_AVG_TIME, 
                y=config.COL_AVG_SCORE, 
                color=config.COL_GROUP if config.COL_GROUP in df.columns else None,
                hover_data=[config.COL_STUDENT_ID, config.COL_WEAKNESS],
                title="學生學習狀態分佈"
            )
            
            # 畫長條圖 (如果有分群結果)
            if config.COL_GROUP in df.columns:
                fig_bar = px.histogram(df, x=config.COL_GROUP, color=config.COL_WEAKNESS, barmode="group", title="各組弱點分佈")
            else:
                fig_bar = None
                
            return _format_for_table(df), fig_scatter, fig_bar

        def run_ai_process():
            """執行 AI 邏輯"""
            df = dm.load_data()
            processed_df, msg = ai.run_analysis_pipeline(df)
            
            # 儲存結果
            dm.save_results(processed_df)
            
            # 重新畫圖 (帶有分群顏色)
            formatted_df, fig_s, fig_b = load_and_plot()

            # 開放儲存按鈕
            return formatted_df, msg, fig_s, fig_b, gr.update(interactive=True)

        # 按鈕綁定
        btn_load.click(fn=load_and_plot, inputs=None, outputs=[data_display, plot_scatter, plot_bar])
        btn_run_ai.click(fn=run_ai_process, inputs=None, outputs=[data_display, result_log, plot_scatter, plot_bar, btn_save])
        btn_save.click(fn=lambda: "已自動儲存於分析流程中", inputs=None, outputs=result_log)

    return app
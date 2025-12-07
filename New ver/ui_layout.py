import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px

from src.ai_engine import TutorAI
from src.data_manager import DataManager
from src import config

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

dm = DataManager()
ai = TutorAI()


def create_ui():
    with gr.Blocks(title="AI 家教進度分派系統") as app:
        gr.Markdown("# 🎓 AI 輔助家教進度分派系統 (Final Fix)")

        with gr.Tabs():
            with gr.Tab("📊 資料匯入與視覺化"):
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

            with gr.Tab("🤖 AI 分析核心"):
                gr.Markdown("### 執行 AI 演算法")
                btn_run_ai = gr.Button("🚀 開始 AI 訓練與分析", variant="primary")
                result_log = gr.Textbox(label="日誌", interactive=False)

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

        def draw_plots(df: pd.DataFrame):
            cols_to_check = [col for col in [config.COL_ACCURACY, config.COL_AVG_TIME] if col in df.columns]
            df = df.dropna(subset=cols_to_check)

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
                            if len(sub_df) == 0:
                                continue

                            x_vals = sub_df[x_col]
                            y_vals = sub_df[y_col]
                            color = colors[idx]

                            ax.scatter(
                                x_vals,
                                y_vals,
                                label=f"Group {group}",
                                color=color,
                                alpha=0.7,
                                edgecolors="w",
                                s=80,
                            )

                            if len(sub_df) > 1:
                                try:
                                    slope, intercept = np.polyfit(x_vals, y_vals, 1)
                                    x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                                    y_line = slope * x_line + intercept
                                    ax.plot(x_line, y_line, color=color, linestyle="--", linewidth=2, alpha=0.9)
                                except Exception as reg_err:
                                    print(f"Regression Error for group {group}: {reg_err}")
                        ax.legend(title="Groups")
                    else:
                        x_vals = df[x_col]
                        y_vals = df[y_col]
                        ax.scatter(x_vals, y_vals, c="tab:blue", alpha=0.7, edgecolors="w", s=80)

                        if len(df) > 1:
                            slope, intercept = np.polyfit(x_vals, y_vals, 1)
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                            y_line = slope * x_line + intercept
                            ax.plot(x_line, y_line, color="red", linestyle="--", linewidth=2, label="Trend")
                            ax.legend()

                    ax.set_xlabel("平均完成時間 (分)")
                    ax.set_ylabel("正確率 (Accuracy)")
                    ax.set_title("學生學習狀態分佈 (含分組回歸分析)", fontname="Microsoft JhengHei", fontsize=14)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    fig.tight_layout()
                    fig_scatter = fig
                except Exception as exc:
                    print(f"Plot Error: {exc}")

            fig_bar = None
            if not df.empty and config.COL_RECOMMENDED_LEVEL in df.columns:
                fig_bar = px.histogram(
                    df,
                    x=config.COL_RECOMMENDED_LEVEL,
                    title="AI 推薦難度分佈 (3 Levels)",
                    text_auto=True,
                    color=config.COL_RECOMMENDED_LEVEL,
                )

            return fig_scatter, fig_bar

        def load_default_data():
            df = dm.load_data()
            fig_s, fig_b = draw_plots(df)
            return df, fig_s, fig_b, "已重置為模擬資料 (空值已濾除)"

        def process_uploaded_file(file_obj):
            if file_obj is None:
                return None, None, None, "請先選擇檔案"
            file_path = file_obj.name if hasattr(file_obj, "name") else file_obj

            df = dm.load_uploaded_file(file_path)

            if df.empty:
                return None, None, None, "檔案為空或欄位不符"

            if config.COL_ACCURACY not in df.columns:
                return df, None, None, f"❌ 錯誤：缺少 '{config.COL_ACCURACY}' 欄位"

            df, log = ai.run_inference_only(df)
            fig_s, fig_b = draw_plots(df)
            return df, fig_s, fig_b, log

        def run_training_pipeline():
            df = dm.load_data()
            processed_df, msg = ai.run_analysis_pipeline(df)
            dm.save_results(processed_df)
            fig_s, fig_b = draw_plots(processed_df)
            return processed_df, msg, fig_s, fig_b

        def predict_user(acc, time, pace, att, hw, e_read, e_vocab, e_logic, m_score):
            result = str(ai.predict_single(acc, time, pace, att, hw, e_read, e_vocab, e_logic, m_score))
            advice = ""
            if "Hard" in result:
                advice = "💡 表現優異，建議挑戰進階課程。"
            elif "Easy" in result:
                advice = "💡 建議加強基礎練習。"
            elif "Medium" in result:
                advice = "💡 進度穩定，適合標準課程。"
            else:
                advice = f"預測結果：{result}"
            return f"# 🎯 結果：{result}", advice

        btn_load.click(fn=load_default_data, outputs=[data_display, plot_scatter, plot_bar, import_log])
        btn_analyze_file.click(
            fn=process_uploaded_file,
            inputs=[file_input],
            outputs=[data_display, plot_scatter, plot_bar, import_log],
        )
        btn_run_ai.click(fn=run_training_pipeline, outputs=[data_display, result_log, plot_scatter, plot_bar])
        btn_predict_user.click(
            fn=predict_user,
            inputs=[
                in_accuracy,
                in_time,
                in_pace,
                in_attend,
                in_hw,
                in_err_read,
                in_err_vocab,
                in_err_logic,
                in_mean_score,
            ],
            outputs=[output_result, output_advice],
        )

        app.load(fn=load_default_data, outputs=[data_display, plot_scatter, plot_bar, import_log])

    return app
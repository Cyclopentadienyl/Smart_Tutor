import networkx as nx
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from src import config

class CurriculumScheduler:
    def __init__(self):
        # 定義課程地圖 (DAG: Directed Acyclic Graph)
        # deps: 前置課程 ID
        # base_hours: 標準學習時數
        # tags: 用於匹配學生弱點
        self.curriculum = {
            "M101": {"name": "M101-基礎代數", "tags": ["logic"], "base_hours": 3.0, "deps": []},
            "M102": {"name": "M102-幾何圖形", "tags": ["visual", "logic"], "base_hours": 4.0, "deps": ["M101"]},
            "E201": {"name": "E201-閱讀理解 A", "tags": ["reading", "vocab"], "base_hours": 2.5, "deps": []},
            "E202": {"name": "E202-詞彙解析", "tags": ["vocab"], "base_hours": 2.0, "deps": ["E201"]},
            "M103": {"name": "M103-進階應用題", "tags": ["logic", "reading"], "base_hours": 5.0, "deps": ["M101", "E201"]},
            "S301": {"name": "S301-科學邏輯", "tags": ["logic", "data"], "base_hours": 3.5, "deps": ["M103"]},
            "FINAL": {"name": "FINAL-期末綜合評量", "tags": ["hard"], "base_hours": 2.0, "deps": ["M102", "E202", "S301"]},
        }

    def _calculate_dynamic_duration(self, chapter_id, student_row):
        """AI 核心：根據學生錯誤率與學習步調，動態調整工時"""
        info = self.curriculum[chapter_id]
        base = info["base_hours"]
        tags = info["tags"]
        
        # 讀取學生特徵
        pace = float(student_row.get(config.COL_LEARNING_PACE, 10.0))
        # 簡單正規化：pace 越小越快 (假設 pace 是 "每題秒數")
        # 這裡假設 pace=10 是標準，pace=20 是慢，pace=5 是快
        pace_modifier = pace / 10.0
        
        skill_modifier = 1.0
        # 讀取錯誤特徵 (從 DataManager 清洗過的欄位)
        err_read = float(student_row.get("err_reading", 0))
        err_logic = float(student_row.get("err_logic", 0))
        err_vocab = float(student_row.get("err_vocab", 0))

        # 如果該章節包含學生常錯的類型，增加工時 (鷹架理論)
        if "logic" in tags and err_logic > 2:
            skill_modifier += 0.3
        if "reading" in tags and err_read > 2:
            skill_modifier += 0.25
        if "vocab" in tags and err_vocab > 2:
            skill_modifier += 0.2
            
        # 如果學生是學霸 (Group = Hard)，整體加速
        group = str(student_row.get(config.COL_GROUP, ""))
        if "Hard" in group:
            skill_modifier -= 0.2

        final_hours = base * pace_modifier * skill_modifier
        return round(max(0.5, final_hours), 1) # 最少 0.5 小時

    def generate_student_schedule(self, student_row, start_date=None):
        """生成個人的甘特圖數據"""
        if start_date is None:
            start_date = datetime.now()

        G = nx.DiGraph()
        
        # 1. 建構圖並計算權重
        for cid, info in self.curriculum.items():
            duration = self._calculate_dynamic_duration(cid, student_row)
            G.add_node(cid, duration=duration, info=info)
            for dep in info["deps"]:
                G.add_edge(dep, cid)

        # 2. 拓撲排序 (確保先修課排在前面)
        try:
            sorted_nodes = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            return pd.DataFrame(), "❌ 課程依賴成環 (Circular Dependency)"

        # 3. 計算時間軸
        schedule_data = []
        # 簡單模擬：假設只有一條執行緒 (Sequential Learning)
        # 若要更高級，可以計算 Critical Path
        current_time = start_date

        for cid in sorted_nodes:
            node = G.nodes[cid]
            duration = node["duration"]
            info = node["info"]
            
            # 假設每天讀書 2 小時，換算天數
            days_needed = duration / 2.0
            end_time = current_time + timedelta(days=days_needed)
            
            schedule_data.append({
                "Task": info["name"],
                "Start": current_time.strftime("%Y-%m-%d"),
                "Finish": end_time.strftime("%Y-%m-%d"),
                "Duration (Hrs)": duration,
                "Type": info["tags"][0], # 取第一個 tag 當顏色分類
                "Resource": student_row.get(config.COL_NAME, "Student")
            })
            
            current_time = end_time # 下一個任務接著做

        return pd.DataFrame(schedule_data), "✅ 路徑規劃完成"

    def plot_gantt(self, df):
        if df.empty:
            return None
        
        fig = px.timeline(
            df, 
            x_start="Start", 
            x_end="Finish", 
            y="Task", 
            color="Type",
            hover_data=["Duration (Hrs)", "Resource"],
            title=f"📅 智慧學習路徑預覽 ({df.iloc[0]['Resource']})",
            height=350
        )
        fig.update_yaxes(autorange="reversed") # 讓開始的任務在最上面
        return fig
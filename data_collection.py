"""
星际争霸2 比赛数据采集脚本
使用Liquipedia API获取2025-2026年比赛数据
"""

import requests
import pandas as pd
import time
import json

# ============================================================
# 1. API配置
# ============================================================

API_KEY = ""
BASE_URL = "https://api.liquipedia.net/api/v3/match"

headers = {
    "Authorization": f"Apikey {API_KEY}",
    "Accept-Encoding": "gzip"
}

params = {
    "wiki": "starcraft2",
    "limit": 1000,
    "query": "date, winner, match2opponents"
}


# ============================================================
# 2. 批量获取比赛数据
# ============================================================

def fetch_all_matches(max_offset=10000):
    """批量获取比赛数据"""
    all_results = []
    offset = 0

    print("=" * 60)
    print("开始采集星际争霸2比赛数据")
    print("=" * 60)

    while offset <= max_offset:
        params["offset"] = offset
        print(f"正在获取 offset={offset}...")

        try:
            r = requests.get(BASE_URL, headers=headers, params=params, timeout=30)

            if r.status_code != 200:
                print(f"❌ API错误: {r.status_code}")
                if r.status_code == 429:
                    print("  请求过于频繁，等待10秒...")
                    time.sleep(10)
                    continue
                break

            data = r.json()
            batch = data.get("result", [])

            if not batch:
                print("✔ 没有更多数据，停止采集")
                break

            all_results.extend(batch)
            print(f"✔ 获取 {len(batch)} 条，累计 {len(all_results)} 条")

            offset += 1000
            time.sleep(1)  # 避免请求过快

        except Exception as e:
            print(f"⚠️ 网络错误: {e}")
            break

    print(f"\n📊 共获取 {len(all_results)} 条比赛记录")
    return all_results


# ============================================================
# 3. 解析对手信息（提取玩家名和种族）
# ============================================================

def parse_opponents(match_data):
    """
    从match2opponents中提取：
    - player1, player2（玩家名）
    - race1, race2（种族：从extradata.faction中获取）
    """
    try:
        opps = match_data.get("match2opponents")

        if not isinstance(opps, list) or len(opps) < 2:
            return None, None, None, None

        # 提取第一个对手
        opp1 = opps[0]
        player1 = opp1.get("name")

        # 从extradata中获取种族（faction）
        race1 = None
        if isinstance(opp1.get("match2players"), list) and len(opp1["match2players"]) > 0:
            extradata = opp1["match2players"][0].get("extradata", {})
            race1 = extradata.get("faction")  # 会返回 't', 'z', 'p'

        # 提取第二个对手
        opp2 = opps[1]
        player2 = opp2.get("name")

        race2 = None
        if isinstance(opp2.get("match2players"), list) and len(opp2["match2players"]) > 0:
            extradata = opp2["match2players"][0].get("extradata", {})
            race2 = extradata.get("faction")

        # 只有当都有玩家名和种族时才返回
        if player1 and player2 and race1 and race2:
            # 将种族代码转换为大写：t->T, z->Z, p->P
            return player1, race1.upper(), player2, race2.upper()

        return None, None, None, None

    except Exception as e:
        return None, None, None, None


# ============================================================
# 4. 解析比赛数据
# ============================================================

def parse_matches(all_results):
    """解析所有比赛数据"""
    print("\n🔍 解析对手信息...")

    parsed_list = []
    for match in all_results:
        p1, r1, p2, r2 = parse_opponents(match)
        parsed_list.append({
            'player1': p1,
            'race1': r1,
            'player2': p2,
            'race2': r2,
            'date': match.get("date"),
            'winner': match.get("winner"),
            'match2id': match.get("match2id")
        })

    df = pd.DataFrame(parsed_list)
    print(f"📊 初始行数: {len(df)}")
    print(f"📊 非空player1: {df['player1'].notna().sum()}")

    return df


# ============================================================
# 5. 数据清洗
# ============================================================

def clean_data(df):
    """清洗数据"""
    print("\n🧹 清洗数据...")

    # 去除有空值的行
    df = df.dropna(subset=["player1", "player2", "race1", "race2", "winner"])
    print(f"📊 去空值后: {len(df)}")

    if len(df) == 0:
        print("❌ 清洗后没有数据了")
        return df

    # 只保留有效种族 (T, Z, P)
    df = df[df["race1"].isin(["T", "Z", "P"]) & df["race2"].isin(["T", "Z", "P"])]
    print(f"📊 种族过滤后: {len(df)}")

    # 去除同族对战（只保留跨种族对战）
    df = df[df["race1"] != df["race2"]]
    print(f"📊 去同族后: {len(df)}")

    # 解析日期
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    print(f"📊 日期解析后: {len(df)}")

    if len(df) == 0:
        print("❌ 清洗后没有数据了")
        return df

    print("\n📅 时间范围：")
    print(f"  最早: {df['date'].min()}")
    print(f"  最晚: {df['date'].max()}")

    return df


# ============================================================
# 6. 筛选2025-2026年数据
# ============================================================

def filter_by_year(df, year_start=2025, year_end=2026):
    """按年份筛选数据"""
    df_filtered = df[
        (df["date"] >= f"{year_start}-01-01") &
        (df["date"] <= f"{year_end}-12-31")
        ]

    if len(df_filtered) == 0:
        print(f"\n⚠️ 没有{year_start}-{year_end}年数据，使用全部数据")
        return df
    else:
        print(f"\n✅ 找到{year_start}-{year_end}年数据: {len(df_filtered)}条")
        return df_filtered


# ============================================================
# 7. 添加辅助列（用于分析）
# ============================================================

def add_analysis_columns(df):
    """添加分析用辅助列"""
    # 胜方标识：winner="1"表示player1获胜
    df["A_win"] = (df["winner"] == "1").astype(int)

    # 对阵标识（排序后，如 P+T, P+Z, T+Z）
    df["matchup"] = df.apply(
        lambda x: "".join(sorted([x["race1"], x["race2"]])),
        axis=1
    )

    # 转换为标准种族名称（方便展示）
    race_full = {"T": "Terran", "Z": "Zerg", "P": "Protoss"}
    df["race1_full"] = df["race1"].map(race_full)
    df["race2_full"] = df["race2"].map(race_full)

    return df


# ============================================================
# 8. 统计分析
# ============================================================

def simple_analysis(df):
    """进行简单的统计分析"""
    print("\n" + "=" * 60)
    print("统计分析结果")
    print("=" * 60)

    # 计算各种族的胜率
    for race, race_name in [("P", "Protoss"), ("T", "Terran"), ("Z", "Zerg")]:
        # 作为player1的胜场
        as_p1 = df[(df["race1"] == race) & (df["A_win"] == 1)].shape[0]
        total_p1 = df[df["race1"] == race].shape[0]

        # 作为player2的胜场
        as_p2 = df[(df["race2"] == race) & (df["A_win"] == 0)].shape[0]
        total_p2 = df[df["race2"] == race].shape[0]

        total_wins = as_p1 + as_p2
        total_games = total_p1 + total_p2

        if total_games > 0:
            winrate = total_wins / total_games
            print(f"\n{race_name}:")
            print(f"  比赛场次: {total_games}")
            print(f"  胜场: {total_wins}")
            print(f"  胜率: {winrate:.2%}")

    # 对阵统计
    print("\n对阵统计:")
    for matchup in ["PT", "PZ", "TZ"]:
        df_matchup = df[df["matchup"] == matchup]
        if len(df_matchup) > 0:
            # 确定哪个种族在前（按字母顺序）
            if matchup == "PT":
                wins_p = df_matchup[
                    ((df_matchup["race1"] == "P") & (df_matchup["A_win"] == 1)) |
                    ((df_matchup["race2"] == "P") & (df_matchup["A_win"] == 0))
                    ].shape[0]
                wins_t = len(df_matchup) - wins_p
                print(
                    f"  {matchup}: P={wins_p}({wins_p / len(df_matchup) * 100:.1f}%), T={wins_t}({wins_t / len(df_matchup) * 100:.1f}%)")
            elif matchup == "PZ":
                wins_p = df_matchup[
                    ((df_matchup["race1"] == "P") & (df_matchup["A_win"] == 1)) |
                    ((df_matchup["race2"] == "P") & (df_matchup["A_win"] == 0))
                    ].shape[0]
                wins_z = len(df_matchup) - wins_p
                print(
                    f"  {matchup}: P={wins_p}({wins_p / len(df_matchup) * 100:.1f}%), Z={wins_z}({wins_z / len(df_matchup) * 100:.1f}%)")
            else:  # TZ
                wins_t = df_matchup[
                    ((df_matchup["race1"] == "T") & (df_matchup["A_win"] == 1)) |
                    ((df_matchup["race2"] == "T") & (df_matchup["A_win"] == 0))
                    ].shape[0]
                wins_z = len(df_matchup) - wins_t
                print(
                    f"  {matchup}: T={wins_t}({wins_t / len(df_matchup) * 100:.1f}%), Z={wins_z}({wins_z / len(df_matchup) * 100:.1f}%)")


# ============================================================
# 9. 保存数据
# ============================================================

def save_data(df, filename="sc2_data.xlsx"):
    """保存数据到Excel"""
    df.to_excel(filename, index=False)
    print(f"\n✅ 导出完成：{filename}")
    print(f"  列: {df.columns.tolist()}")
    print(f"  行数: {len(df)}")


# ============================================================
# 10. 主函数
# ============================================================

def main():
    """主函数"""
    # 1. 获取数据
    all_results = fetch_all_matches(max_offset=10000)

    if len(all_results) == 0:
        print("❌ API没有返回数据")
        return

    # 2. 解析数据
    df = parse_matches(all_results)

    # 3. 清洗数据
    df = clean_data(df)

    if len(df) == 0:
        print("❌ 数据清洗后无有效记录")
        return

    # 4. 筛选2025-2026年数据
    df_final = filter_by_year(df, year_start=2025, year_end=2026)

    # 5. 添加分析列
    df_final = add_analysis_columns(df_final)

    # 6. 统计分析
    simple_analysis(df_final)

    # 7. 保存数据
    save_data(df_final, "sc2_data_2025_2026.xlsx")

    print("\n" + "=" * 60)
    print("数据采集完成！")
    print("=" * 60)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()
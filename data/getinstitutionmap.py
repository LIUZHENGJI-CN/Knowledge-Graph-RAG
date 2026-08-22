import requests
import json
import os

def expand_institution_map():
    # 1. 定义本地文件和远程数据源
    MAP_FILE = "C:/Users/lzj/Desktop/申请博士/复现知识图谱/data/institution_map.json"
    GITHUB_RAW_URL = "https://raw.githubusercontent.com/Hipo/university-domains-list/master/world_universities_and_domains.json"
    
    # 2. 初始核心映射 (您之前提供的核心名单)
    final_map = {
        "fudan.edu.cn": "Fudan University",
        "pku.edu.cn": "Peking University",
        "tsinghua.edu.cn": "Tsinghua University",
        "sjtu.edu.cn": "Shanghai Jiao Tong University",
        "zju.edu.cn": "Zhejiang University"
    }

    print(f"--- 正在从 GitHub 获取全球大学数据... ---")
    try:
        # 3. 获取 GitHub 上的开源数据
        response = requests.get(GITHUB_RAW_URL, timeout=10)
        response.raise_for_status()
        world_data = response.json()
        
        # 4. 转换数据格式：该项目格式为 [{"name": "...", "domains": ["..."]}, ...]
        new_entries_count = 0
        for uni in world_data:
            name = uni.get("name")
            domains = uni.get("domains", [])
            for domain in domains:
                if domain not in final_map:
                    final_map[domain] = name
                    new_entries_count += 1
        
        # 5. 保存到本地 JSON 文件
        with open(MAP_FILE, "w", encoding="utf-8") as f:
            json.dump(final_map, f, indent=4, ensure_ascii=False)
            
        print(f"--- 扩充完成！ ---")
        print(f"共计导入 {new_entries_count} 条新对应关系。")
        print(f"当前 institution_map.json 总计包含 {len(final_map)} 条记录。")

    except Exception as e:
        print(f"获取失败: {e}")
        print("请检查网络连接或手动从 GitHub 下载数据。")

if __name__ == "__main__":
    expand_institution_map()

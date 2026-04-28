# AR 问答引文追踪示例

## 完整的 SSE 流示例

当 AR 设备调用 `/ar/chat` 时，完整的流式响应如下：

```
event: citations
data: {"items": [...], "relations": [...]}

event: message
data: {"content": "斗栱"}

event: message
data: {"content": "是传统木构建筑中"}

...

event: done
data: {"status": "finished"}
```

---

## 详细数据结构

### 1️⃣ `citations` 事件 - 引文信息

完整示例（真实数据）：

```json
{
  "event": "citations",
  "data": {
    "items": [
      {
        "id": "chunk_001_text",
        "type": "text",
        "content": "斗栱（douˇgǒng）：木构建筑中用来承托屋顶、传递载荷的构件组合。由方形的'斗'和弧形的'栱'组成，是传统建筑的标志性构件。",
        "metadata": {
          "book_id": "yingzaofashi_v1",
          "book_name": "营造法式",
          "content_type": "original_text",
          "closest_title": "卷一 总释名数",
          "toc_path": ["卷一", "总释名数", "构件名称"],
          "chunk_size": 128
        },
        "score": 0.9876,
        "is_main": true
      },
      {
        "id": "chunk_042_text",
        "type": "text",
        "content": "斗栱的分类依其位置和功能分为：柱头斗栱（支撑梁头）、转角斗栱（处理转角压力）、补间斗栱（补充柱间距离）。",
        "metadata": {
          "book_id": "yingzaofashi_v1",
          "book_name": "营造法式",
          "content_type": "original_text",
          "closest_title": "卷三 斗栱制度",
          "toc_path": ["卷三", "斗栱制度", "斗栱分类"],
          "chunk_size": 96
        },
        "score": 0.8543,
        "is_main": true
      },
      {
        "id": "chunk_156_image",
        "type": "image",
        "content": "斗栱结构示意图",
        "metadata": {
          "book_id": "yingzaofashi_v1",
          "book_name": "营造法式",
          "title": "斗栱结构示意图",
          "image_uri": "https://example.com/images/dougong_structure.jpg",
          "local_path": "/data/images/dougong_structure.jpg",
          "alt_text": "展示斗、栱、垫板等构件如何组合的示意图",
          "closest_title": "卷三 斗栱制度",
          "toc_path": ["卷三", "斗栱制度", "构件图示"],
          "format": "jpg"
        },
        "score": 0.7812,
        "is_main": true
      }
    ],
    "relations": [
      {
        "source_id": "chunk_001_text",
        "target_id": "chunk_042_text",
        "relation_type": "illustrates"
      },
      {
        "source_id": "chunk_042_text",
        "target_id": "chunk_156_image",
        "relation_type": "illustrates"
      }
    ]
  }
}
```

---

## 数据字段详解

### `items` 数组 - 各个文献段落

| 字段 | 类型 | 说明 | 示例 |
|------|------|------|------|
| **id** | string | 唯一标识 | `chunk_001_text` |
| **type** | string | 内容类型：`text` 或 `image` | `text` |
| **content** | string | 实际内容 | 文献文本或图片描述 |
| **score** | float | 相关性得分 (0-1) | `0.9876` |
| **is_main** | bool | 是否为主要结果 | `true` |
| **metadata** | object | 元数据（见下表） | {...} |

### `metadata` 对象 - 文献元信息

#### 对于 **text** 类型：

| 字段 | 说明 | 示例 |
|------|------|------|
| `book_id` | 所属书籍 ID | `yingzaofashi_v1` |
| `book_name` | 书籍名称 | `营造法式` |
| `content_type` | 内容类型 | `original_text`, `annotation`, `modern_translation`, `interpretation`, `others_text` |
| `closest_title` | 最近的章节标题 | `卷一 总释名数` |
| `toc_path` | 目录路径 | `["卷一", "总释名数", "构件名称"]` |
| `chunk_size` | 段落大小（字数）| `128` |
| `other_metadata` | 其他自定义元数据 | 可选 |

#### 对于 **image** 类型：

| 字段 | 说明 | 示例 |
|------|------|------|
| `book_id` | 所属书籍 ID | `yingzaofashi_v1` |
| `book_name` | 书籍名称 | `营造法式` |
| `title` | 图片标题 | `斗栱结构示意图` |
| `image_uri` | 外网 URL（若有） | `https://example.com/...` |
| `local_path` | 本地路径 | `/data/images/...` |
| `alt_text` | 图片描述 | 屏幕阅读器用 |
| `closest_title` | 最近的章节标题 | `卷三 斗栱制度` |
| `toc_path` | 目录路径 | `["卷三", "斗栱制度", "构件图示"]` |
| `format` | 文件格式 | `jpg`, `png` |

### `relations` 数组 - 文献关系

| 字段 | 说明 | 示例 |
|------|------|------|
| `source_id` | 源文献 ID | `chunk_001_text` |
| `target_id` | 目标文献 ID | `chunk_042_text` |
| `relation_type` | 关系类型 | `illustrates`（阐述）, `annotates`（注释） |

---

## Unity 中的使用示例

### C# 数据结构定义

```csharp
[System.Serializable]
public class Citation
{
    public string id;
    public string type;  // "text" or "image"
    public string content;
    public Metadata metadata;
    public float score;
    public bool is_main;
}

[System.Serializable]
public class Metadata
{
    public string book_id;
    public string book_name;
    
    // 对于 text 类型
    public string content_type;
    public string closest_title;
    public string[] toc_path;
    public int chunk_size;
    
    // 对于 image 类型
    public string title;
    public string image_uri;
    public string local_path;
    public string alt_text;
    public string format;
}

[System.Serializable]
public class Relation
{
    public string source_id;
    public string target_id;
    public string relation_type;  // "illustrates" or "annotates"
}

[System.Serializable]
public class CitationData
{
    public Citation[] items;
    public Relation[] relations;
}
```

### 处理 citations 事件

```csharp
private void HandleCitations(string json)
{
    CitationData citations = JsonUtility.FromJson<CitationData>(json);
    
    Debug.Log($"找到 {citations.items.Length} 条引文");
    
    // 遍历文献
    foreach (var item in citations.items)
    {
        if (item.type == "text")
        {
            Debug.Log($"[{item.id}] {item.metadata.book_name} > {string.Join(" > ", item.metadata.toc_path)}");
            Debug.Log($"内容类型: {item.metadata.content_type}");
            Debug.Log($"相关性: {item.score:P2}");  // 99.76%
            Debug.Log($"内容预览: {item.content.Substring(0, Math.Min(100, item.content.Length))}...");
        }
        else if (item.type == "image")
        {
            Debug.Log($"[{item.id}] 图片: {item.metadata.title}");
            Debug.Log($"地址: {item.metadata.local_path}");
            // TODO: 加载图片显示
            LoadImageUI(item.metadata.local_path, item.metadata.title);
        }
    }
    
    // 处理文献关系
    foreach (var relation in citations.relations)
    {
        Debug.Log($"{relation.source_id} --{relation.relation_type}--> {relation.target_id}");
    }
}

private void LoadImageUI(string imagePath, string title)
{
    // 加载本地图片或远程图片
    // 显示在 AR UI 中
    Debug.Log($"加载图片: {imagePath}");
}
```

---

## 实际调用流程示例

### 用户问题
```
"斗栱是什么？"
```

### SSE 响应流

```
1️⃣ 立即返回：citations 事件
event: citations
data: {"items": [{"id": "chunk_001_text", "type": "text", ...}], "relations": []}

2️⃣ 然后流式返回：message 事件（多个）
event: message
data: {"content": "斗"}

event: message
data: {"content": "栱"}

event: message
data: {"content": "是传统木构建筑中用来承托屋顶、传递载荷的构件组合。"}

...

3️⃣ 最后返回：done 事件
event: done
data: {"status": "finished"}
```

---

## 显示引文的 UI 示例（伪代码）

```csharp
public void DisplayAnswer(CitationData citations, string answer)
{
    // 左侧：答案文本，点击可跳转到引文
    DisplayAnswerText(answer, citations);
    
    // 右侧：引文卡片堆栈
    foreach (var item in citations.items)
    {
        var card = new CitationCard();
        
        if (item.type == "text")
        {
            card.SetTitle($"{item.metadata.book_name}");
            card.SetSubtitle(string.Join(" > ", item.metadata.toc_path));
            card.SetType(GetContentTypeLabel(item.metadata.content_type));
            card.SetScore($"相关性: {item.score:P1}");
            card.SetContent(item.content);
        }
        else
        {
            card.SetTitle($"图片: {item.metadata.title}");
            card.SetImage(item.metadata.local_path);
            card.SetAlt(item.metadata.alt_text);
        }
        
        citationPanel.AddCard(card);
    }
}

private string GetContentTypeLabel(string contentType)
{
    return contentType switch
    {
        "original_text" => "原文",
        "annotation" => "注释",
        "modern_translation" => "译文",
        "interpretation" => "解读",
        _ => "其他"
    };
}
```

---

## 关键特性

✅ **实时引文追踪** - 答案生成时同步推送引文  
✅ **来源溯源** - 精确到卷、章、节  
✅ **多媒体支持** - 文本 + 图片引文  
✅ **相关性评分** - 知道哪些引文最相关  
✅ **文献关系** - 了解各文献之间的联系  

---

## 常见问题

**Q: 为什么有些答案没有引文？**  
A: Gate 判定问题不需要检索（如闲聊）或知识库未覆盖时，`skip_rag=true`，citations 为空。

**Q: 相关性分数是怎么算的？**  
A: 使用 RRF (Reciprocal Rank Fusion) 融合向量搜索 + 关键词搜索的排名，值范围 0-1。

**Q: 为什么同一问题的引文每次不同？**  
A: 每次都是独立调用，检索结果可能略有差异，但核心引文通常相同。

**Q: 可以自定义 items 返回数量吗？**  
A: 目前固定为 top-10，若需修改可调整 `/ar/chat` 实现或在 Agent 配置中调整。

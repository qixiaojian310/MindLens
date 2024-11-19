---
title: 个人项目
language_tabs:
  - shell: Shell
  - http: HTTP
  - javascript: JavaScript
  - ruby: Ruby
  - python: Python
  - php: PHP
  - java: Java
  - go: Go
toc_footers: []
includes: []
search: true
code_clipboard: true
highlight_theme: darkula
headingLevel: 2
generator: "@tarslib/widdershins v4.0.23"

---

# 个人项目

Base URLs:

# Authentication

# Default

## POST 获取预测

POST /getPrediction

> Body 请求参数

```yaml
features: ""

```

### 请求参数

|名称|位置|类型|必选|说明|
|---|---|---|---|---|
|body|body|object| 否 |none|
|» features|body|[boolean]| 否 |症状：使用map传输|

> 返回示例

> 200 Response

```json
{
  "PTSD": 0,
  "BipolarDisorder": 0,
  "Depression": 0,
  "AnxietyDisorder": 0,
  "Schizophrenia": 0
}
```

### 返回结果

|状态码|状态码含义|说明|数据模型|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|none|Inline|
|500|[Internal Server Error](https://tools.ietf.org/html/rfc7231#section-6.6.1)|none|Inline|

### 返回数据结构

状态码 **200**

*不同疾病的概率*

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|» PTSD|number|true|none||none|
|» BipolarDisorder|number|true|none||none|
|» Depression|number|true|none||none|
|» AnxietyDisorder|number|true|none||none|
|» Schizophrenia|number|true|none||none|

状态码 **500**

|名称|类型|必选|约束|中文名|说明|
|---|---|---|---|---|---|
|» ErrorCode|integer|true|none||none|

#### 枚举值

|属性|值|
|---|---|
|ErrorCode|1|

# 数据模型


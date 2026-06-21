# PROJECT_MEMORY.md

## 项目概况
- 项目名：`zcode2api`
- 本地路径：`/opt/data/workspace/zcode2api`
- GitHub 仓库：`https://github.com/Roiung/zcode2api`
- 当前分支：`master`
- 当前状态：已拉取到本地，工作区干净，可直接进入修改阶段。

## 当前阶段
- 已完成本地接管与初步代码审查。
- 已完成网关兼容性改造：补齐 CORS / OPTIONS 支持，并新增 OpenAI Chat Completions 兼容接口。
- OpenAI 兼容层已进一步补齐 `tools / tool_choice / tool_calls` 的基础协议映射。
- 现已新增 `/v1/responses` 兼容接口，支持新版 OpenAI Responses API 的基础接入。
- 已完成一次全面代码核查，并修复兼容接口可能绕过网关 API Key 鉴权的缺陷。
- 当前代码已进入“可供 Anthropic / OpenAI 风格客户端接入验证”的阶段。
- 后续重点应放在带真实账号的端到端联调与字段兼容细化，而不是重新搭项目。

## 当前已确认的架构结论
- 这是一个 **Python/FastAPI 单体网关服务**，不是前后端分离项目。
- 核心目标：将 `ZCode / Z.AI` 能力封装为兼容 `Anthropic Messages API` 的 `/v1/messages` 服务。
- 现已额外暴露 `OpenAI Chat Completions` 兼容接口 `/v1/chat/completions`，内部仍复用 Anthropic 主链路。
- 项目同时包含：
  - 网关 API
  - 后台管理 UI
  - 多账号轮询与状态管理
  - SQLite 本地持久化
  - JWT / API Key 两类账号模式
  - 阿里云无痕验证码求解（Node + jsdom 子进程）
  - OAuth 登录与额度监控

## 关键文件地图
- CLI 入口：`main.py`
- FastAPI 应用装配：`app/main.py`
- 网关主逻辑：`app/routes/gateway.py`
- 后台 API：`app/routes/admin_api.py`
- 页面路由：`app/routes/pages.py`
- 存储与轮询：`app/store.py`
- 账号模型：`app/models.py`
- 上游请求构建：`app/agent.py`
- 验证码管理：`app/captcha.py`
- 额度刷新：`app/quota.py`
- OAuth：`app/oauth.py`
- 架构文档：`docs/ARCHITECTURE.md`

## 当前理解的主控制流
1. 客户端请求进入 `/v1/messages`
2. 网关根据模型名 / header 判定 provider，并规范化请求体
3. 从账号池按 round-robin 选择可用账号
4. 若为 `zai + jwt` 模式，则先获取验证码 `verifyParam`
5. 构建上游请求并转发
6. 根据返回结果更新账号状态：`active / exhausted / cooling / invalid / disabled`
7. 成功时透传流式响应，并异步刷新额度信息

## 当前已识别的设计特点
- 结构清晰，属于轻量但完整的单体服务，适合快速定点修改。
- `store / gateway / captcha / quota` 是后续改造最核心的四个模块。
- 后台 UI 为内嵌静态页面，部署链路简单。
- `docs/ARCHITECTURE.md` 与当前代码整体一致，可作为后续改造参考。

## 当前已识别的风险点
- 错误切换逻辑部分依赖关键字匹配，后续若上游返回格式变化，可能误判。
- 账号状态与轮询游标主要维护在单进程内存中，天然偏单实例架构。
- 验证码求解依赖阿里云 SDK 现状，是最敏感、最容易受上游变化影响的模块。
- 若后续要改后台 UI，还需继续细读 `app/statics/admin/*` 页面与脚本。

## 后续改造建议
- 改 API 行为：优先看 `app/routes/gateway.py`
- 改账号池 / 密钥 / 状态：优先看 `app/store.py` 与 `app/models.py`
- 改稳定性 / 验证码 / 限流恢复：优先看 `app/captcha.py` 与 `app/quota.py`
- 改后台管理交互：继续分析 `app/statics/admin/*`

## 备注
- 当前只是完成“接手仓库 + 初步分析”，尚未对代码做任何功能性修改。
- 后续用户一旦明确改造目标，应直接在该仓库上实施，不再重复做架构摸底。

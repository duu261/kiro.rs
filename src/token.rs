//! Token 计算模块
//!
//! 提供文本 token 数量计算功能。
//!
//! # 计算规则
//! - 非西文字符：每个计 4.5 个字符单位
//! - 西文字符：每个计 1 个字符单位
//! - 4 个字符单位 = 1 token（四舍五入）

use crate::anthropic::types::{
    CountTokensRequest, CountTokensResponse, Message, SystemMessage, Tool,
};
use crate::http_client::{ProxyConfig, build_client};
use crate::model::config::TlsBackend;
use std::sync::OnceLock;

/// Count Tokens API 配置
#[derive(Clone, Default)]
pub struct CountTokensConfig {
    /// 外部 count_tokens API 地址
    pub api_url: Option<String>,
    /// count_tokens API 密钥
    pub api_key: Option<String>,
    /// count_tokens API 认证类型（"x-api-key" 或 "bearer"）
    pub auth_type: String,
    /// 代理配置
    pub proxy: Option<ProxyConfig>,

    pub tls_backend: TlsBackend,
}

/// 全局配置存储
static COUNT_TOKENS_CONFIG: OnceLock<CountTokensConfig> = OnceLock::new();

/// 初始化 count_tokens 配置
///
/// 应在应用启动时调用一次
pub fn init_config(config: CountTokensConfig) {
    let _ = COUNT_TOKENS_CONFIG.set(config);
}

/// 获取配置
fn get_config() -> Option<&'static CountTokensConfig> {
    COUNT_TOKENS_CONFIG.get()
}

/// 判断字符是否为非西文字符
///
/// 西文字符包括：
/// - ASCII 字符 (U+0000..U+007F)
/// - 拉丁字母扩展 (U+0080..U+024F)
/// - 拉丁字母扩展附加 (U+1E00..U+1EFF)
///
/// 返回 true 表示该字符是非西文字符（如中文、日文、韩文、阿拉伯文等）
fn is_non_western_char(c: char) -> bool {
    !matches!(c,
        // 基本 ASCII
        '\u{0000}'..='\u{007F}' |
        // 拉丁字母扩展-A (Latin Extended-A)
        '\u{0080}'..='\u{00FF}' |
        // 拉丁字母扩展-B (Latin Extended-B)
        '\u{0100}'..='\u{024F}' |
        // 拉丁字母扩展附加 (Latin Extended Additional)
        '\u{1E00}'..='\u{1EFF}' |
        // 拉丁字母扩展-C/D/E
        '\u{2C60}'..='\u{2C7F}' |
        '\u{A720}'..='\u{A7FF}' |
        '\u{AB30}'..='\u{AB6F}'
    )
}

/// 计算文本的 token 数量
///
/// # 计算规则
/// - 非西文字符：每个计 4.5 个字符单位
/// - 西文字符：每个计 1 个字符单位
/// - 4 个字符单位 = 1 token（四舍五入）
/// ```
pub fn count_tokens(text: &str) -> u64 {
    // println!("text: {}", text);

    let char_units: f64 = text
        .chars()
        .map(|c| if is_non_western_char(c) { 4.0 } else { 1.0 })
        .sum();

    let tokens = char_units / 4.0;

    let acc_token = if tokens < 100.0 {
        tokens * 1.5
    } else if tokens < 200.0 {
        tokens * 1.3
    } else if tokens < 300.0 {
        tokens * 1.25
    } else if tokens < 800.0 {
        tokens * 1.2
    } else {
        tokens * 1.0
    } as u64;

    // println!("tokens: {}, acc_tokens: {}", tokens, acc_token);
    acc_token
}

/// 估算请求的输入 tokens
///
/// 优先调用远程 API，失败时回退到本地计算
pub(crate) fn count_all_tokens(
    model: String,
    system: Option<Vec<SystemMessage>>,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
) -> u64 {
    // 检查是否配置了远程 API
    if let Some(config) = get_config() {
        if let Some(api_url) = &config.api_url {
            // 尝试调用远程 API
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(call_remote_count_tokens(
                    api_url, config, model, &system, &messages, &tools,
                ))
            });

            match result {
                Ok(tokens) => {
                    tracing::debug!("远程 count_tokens API 返回: {}", tokens);
                    return tokens;
                }
                Err(e) => {
                    tracing::warn!("远程 count_tokens API 调用失败，回退到本地计算: {}", e);
                }
            }
        }
    }

    // 本地计算
    count_all_tokens_local(system, messages, tools)
}

/// 调用远程 count_tokens API
async fn call_remote_count_tokens(
    api_url: &str,
    config: &CountTokensConfig,
    model: String,
    system: &Option<Vec<SystemMessage>>,
    messages: &Vec<Message>,
    tools: &Option<Vec<Tool>>,
) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
    let client = build_client(config.proxy.as_ref(), 300, config.tls_backend)?;

    // 构建请求体
    let request = CountTokensRequest {
        model: model, // 模型名称用于 token 计算
        messages: messages.clone(),
        system: system.clone(),
        tools: tools.clone(),
    };

    // 构建请求
    let mut req_builder = client.post(api_url);

    // 设置认证头
    if let Some(api_key) = &config.api_key {
        if config.auth_type == "bearer" {
            req_builder = req_builder.header("Authorization", format!("Bearer {}", api_key));
        } else {
            req_builder = req_builder.header("x-api-key", api_key);
        }
    }

    // 发送请求
    let response = req_builder
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        return Err(format!("API 返回错误状态: {}", response.status()).into());
    }

    let result: CountTokensResponse = response.json().await?;
    Ok(result.input_tokens as u64)
}

/// 本地计算消息 content 的 tokens（字符串或 block 数组）。
///
/// 覆盖 text / thinking / tool_use.input / tool_result.content（字符串或嵌套 block）/
/// image 以及其他未知 block（按序列化 JSON 计）。仅数 `text` 字段会让 agentic
/// 会话（以 tool_result 为主）的输入被严重低估。
fn count_content_tokens(content: &serde_json::Value) -> u64 {
    match content {
        serde_json::Value::String(s) => count_tokens(s),
        serde_json::Value::Array(arr) => arr.iter().map(count_block_tokens).sum(),
        serde_json::Value::Null => 0,
        other => count_tokens(&other.to_string()),
    }
}

fn count_block_tokens(block: &serde_json::Value) -> u64 {
    let str_field = |key: &str| block.get(key).and_then(|v| v.as_str());
    match block.get("type").and_then(|v| v.as_str()) {
        Some("text") => str_field("text").map(count_tokens).unwrap_or(0),
        Some("thinking") => str_field("thinking").map(count_tokens).unwrap_or(0),
        Some("redacted_thinking") => 8,
        Some("tool_use") => {
            let input = block
                .get("input")
                .map(|v| serde_json::to_string(v).unwrap_or_default())
                .unwrap_or_default();
            str_field("name").map(count_tokens).unwrap_or(0) + count_tokens(&input)
        }
        Some("tool_result") => block
            .get("content")
            .map(count_content_tokens)
            .unwrap_or(0),
        Some("image") => {
            let src = block.get("source");
            let media_type = src
                .and_then(|s| s.get("media_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let data = src
                .and_then(|s| s.get("data"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            crate::image_resize::estimate_image_tokens(media_type, data) as u64
        }
        _ => match str_field("text") {
            Some(text) => count_tokens(text),
            None => count_tokens(&serde_json::to_string(block).unwrap_or_default()),
        },
    }
}

/// 本地计算请求的输入 tokens
fn count_all_tokens_local(
    system: Option<Vec<SystemMessage>>,
    messages: Vec<Message>,
    tools: Option<Vec<Tool>>,
) -> u64 {
    let mut total = 0;

    // 系统消息
    if let Some(ref system) = system {
        for msg in system {
            total += count_tokens(&msg.text);
        }
    }

    // 用户消息
    for msg in &messages {
        total += count_content_tokens(&msg.content);
    }

    // 工具定义
    if let Some(ref tools) = tools {
        for tool in tools {
            total += count_tokens(&tool.name);
            total += count_tokens(&tool.description);
            let input_schema_json = serde_json::to_string(&tool.input_schema).unwrap_or_default();
            total += count_tokens(&input_schema_json);
        }
    }

    total.max(1)
}

/// 估算输出 tokens
pub(crate) fn estimate_output_tokens(content: &[serde_json::Value]) -> i32 {
    let mut total = 0;

    for block in content {
        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
            total += count_tokens(text) as i32;
        }
        if let Some(thinking) = block.get("thinking").and_then(|v| v.as_str()) {
            total += count_tokens(thinking) as i32;
        }
        if block.get("type").and_then(|v| v.as_str()) == Some("redacted_thinking") {
            total += 8;
        }
        if block.get("type").and_then(|v| v.as_str()) == Some("tool_use") {
            // 工具调用开销
            if let Some(input) = block.get("input") {
                let input_str = serde_json::to_string(input).unwrap_or_default();
                total += count_tokens(&input_str) as i32;
            }
        }
    }

    total.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn estimate_output_tokens_counts_thinking_blocks() {
        let with_thinking = estimate_output_tokens(&[json!({
            "type": "thinking",
            "thinking": "需要计入输出 token"
        })]);
        let text_only = estimate_output_tokens(&[json!({
            "type": "text",
            "text": ""
        })]);

        assert!(with_thinking > text_only);
    }

    #[test]
    fn estimate_output_tokens_counts_redacted_thinking() {
        let tokens = estimate_output_tokens(&[json!({
            "type": "redacted_thinking",
            "data": "encrypted"
        })]);

        assert!(tokens >= 8);
    }

    fn msg(role: &str, content: serde_json::Value) -> Message {
        Message {
            role: role.to_string(),
            content,
        }
    }

    #[test]
    fn local_input_counts_tool_result_and_tool_use() {
        let big = "x".repeat(40_000);
        let text_only = count_all_tokens_local(
            None,
            vec![msg("user", json!([{ "type": "text", "text": "run ls" }]))],
            None,
        );
        let agentic = count_all_tokens_local(
            None,
            vec![
                msg("user", json!([{ "type": "text", "text": "run ls" }])),
                msg(
                    "assistant",
                    json!([{ "type": "tool_use", "id": "t1", "name": "shell",
                             "input": { "cmd": big.clone() } }]),
                ),
                msg(
                    "user",
                    json!([{ "type": "tool_result", "tool_use_id": "t1", "content": big.clone() }]),
                ),
                msg(
                    "user",
                    json!([{ "type": "tool_result", "tool_use_id": "t2",
                             "content": [{ "type": "text", "text": big.clone() }] }]),
                ),
            ],
            None,
        );
        // 3 × 40k chars ≈ 30k tokens；旧实现只数 text 字段，约等于 text_only。
        assert!(agentic >= text_only + 29_000, "agentic={agentic} text_only={text_only}");
    }

    #[test]
    fn local_input_grows_with_tool_output() {
        let turn = |n: usize| {
            count_all_tokens_local(
                None,
                vec![msg(
                    "user",
                    json!([{ "type": "tool_result", "tool_use_id": "t", "content": "y".repeat(n) }]),
                )],
                None,
            )
        };
        assert!(turn(80_000) > turn(8_000) * 5);
    }

    #[test]
    fn local_input_counts_thinking() {
        let with_thinking = count_all_tokens_local(
            None,
            vec![msg(
                "assistant",
                json!([{ "type": "thinking", "thinking": "z".repeat(4_000), "signature": "s" }]),
            )],
            None,
        );
        assert!(with_thinking >= 900);
    }
}

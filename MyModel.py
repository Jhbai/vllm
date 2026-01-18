import torch
import gc
import json
import re
from typing import Any, List, Optional, Iterator, Dict, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk, ToolCall
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr, Field
from transformers import (
    Gemma3ForCausalLM,
    GemmaTokenizerFast,
    BitsAndBytesConfig,
    DynamicCache
)

# 使用 allow_tf32：在 Ampere 架構 GPU 上允許 TF32 運算
torch.backends.cuda.matmul.allow_tf32 = True

class TransformersModel(BaseChatModel):
    """
    針對高吞吐量優化的 Hugging Face Chat Model (整合 Gemma 3 手動 Cache 控制 + Tool Calling 支援)。
    """
    model_name: str = Field(..., description="Hugging Face 模型名稱")
    temperature: float = 0.1
    max_new_tokens: int = 1024
    
    _model: Any = PrivateAttr()
    _tokenizer: Any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 1. 量化設定 (與 gemma_llm.py 保持一致)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # 2. 載入 Tokenizer
        self._tokenizer = GemmaTokenizerFast.from_pretrained(self.model_name)
        
        # 3. 載入模型
        self._model = Gemma3ForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        self._model = self._model.eval()

    @property
    def _llm_type(self) -> str:
        return "gemma3_custom_optimized"

    def bind_tools(self, tools: list, **kwargs):
        """
        顯式實作 bind_tools，雖然 BaseChatModel 有，但這裡我們可以做一些預處理或檢查。
        回傳的是 RunnableBinding，LangChain 會在 invoke 時將 tools 放入 kwargs。
        """
        return self.bind(tools=tools, **kwargs)

    def _format_tool_prompt(self, tools: List[Dict]) -> str:
        """將 LangChain tools 轉換為 System Prompt 的一部分"""
        if not tools:
            return ""
        
        tool_descs = []
        for tool in tools:
            # 轉換為 OpenAI 格式的 dict 以便統一處理
            t = convert_to_openai_tool(tool)
            func = t["function"]
            tool_descs.append(f"Name: {func['name']}\nDescription: {func['description']}\nParameters: {json.dumps(func['parameters'])}")
        
        tool_prompt = (
            "\nYou have access to the following tools. "
            "If you need to use a tool, please output ONLY a JSON object with the format: "
            '{"name": "tool_name", "arguments": {"param": "value"}}.\n'
            "Available Tools:\n" + "\n---\n".join(tool_descs) + "\n"
        )
        return tool_prompt

    def _format_messages_to_prompt(self, messages: List[BaseMessage], tools: Optional[List[Dict]] = None) -> str:
        """將 LangChain Messages 轉換為 Prompt，並注入工具說明"""
        hf_messages = []
        system_content = ""
        
        # 1. 提取 System Message
        for m in messages:
            if m.type == "system":
                system_content += m.content + "\n"
        
        # 2. 如果有綁定工具，將工具說明加入 System Prompt
        if tools:
            system_content += self._format_tool_prompt(tools)

        # 3. 處理 System Message (Gemma 通常將 System 視為 User 的第一句話或獨立處理)
        if system_content:
            hf_messages.append({"role": "system", "content": system_content.strip()})

        # 4. 處理其餘訊息
        for m in messages:
            if m.type == "system": 
                continue 
            
            role = "user" if m.type == "human" else "assistant" if m.type == "ai" else "tool"
            content = m.content
            
            if m.type == "tool":
                role = "user"
                content = f"Tool '{m.name}' output: {m.content}"
            
            hf_messages.append({"role": role, "content": content})
            
        # --- 關鍵修正：定義一個針對純文字的 Jinja Template ---
        # 這會覆蓋模型預設可能預期多模態物件的行為
        gemma_chat_template = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
                "{{ '<start_of_turn>user\n' + messages[0]['content'] | trim + '<end_of_turn>\n' }}"
                "{% set loop_messages = messages[1:] %}"
            "{% else %}"
                "{% set loop_messages = messages %}"
            "{% endif %}"
            "{% for message in loop_messages %}"
                "{% if message['role'] == 'user' %}"
                    "{{ '<start_of_turn>user\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                    "{{ '<start_of_turn>model\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                "{% else %}"
                    "{{ '<start_of_turn>' + message['role'] + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "{{ '<start_of_turn>model\n' }}"
            "{% endif %}"
        )

        # 5. 呼叫 apply_chat_template 並傳入 chat_template 參數
        return self._tokenizer.apply_chat_template(
            hf_messages, 
            chat_template=gemma_chat_template, # 強制使用我們定義的 template
            tokenize=False, 
            add_generation_prompt=True
        )

    def _inference_core(self, input_ids: torch.Tensor) -> Iterator[str]:
        """
        核心推論邏輯：完全移植自 gemma_llm.py (保持不變)
        """
        eos_token_ids = [self._tokenizer.eos_token_id, 106] 
        past_key_values = DynamicCache()
        chunks = torch.split(input_ids[:, :-1], 32, dim=-1)
        st = 0
        ed = 0
        
        try:
            with torch.no_grad():
                for chunk in chunks:
                    ed = st + chunk.shape[1]
                    self._model(input_ids=chunk, use_cache=True, past_key_values=past_key_values)
                    st = ed
            
            curr_input_ids = input_ids[:, -1:]
            
            for _ in range(self.max_new_tokens):
                with torch.no_grad():
                    ed += 1
                    seq_len = past_key_values.get_seq_length(layer_idx=0)
                    cache_position = torch.arange(seq_len - 1, seq_len, dtype=torch.long, device=self._model.device)
                    
                    outputs = self._model(
                        input_ids=curr_input_ids, 
                        use_cache=True, 
                        past_key_values=past_key_values, 
                        cache_position=cache_position
                    )
                    
                    logits = outputs.logits
                    next_token_logits = logits[:, -1, :]
                    
                    if self.temperature > 0:
                        probs = torch.softmax(next_token_logits / self.temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    token_id = next_token.item()
                    curr_input_ids = next_token

                    if token_id in eos_token_ids:
                        break

                    token_text = self._tokenizer.decode(token_id)
                    yield token_text
                    
        except KeyboardInterrupt:
            pass
        finally:
            del input_ids, curr_input_ids
            if 'outputs' in locals(): del outputs
            if 'logits' in locals(): del logits
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _parse_tool_call(self, text: str) -> Optional[List[ToolCall]]:
        """簡單的 JSON 解析器，檢查模型是否輸出了工具呼叫"""
        text = text.strip()
        # 嘗試捕捉 JSON 區塊
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                if "name" in data and "arguments" in data:
                    return [ToolCall(
                        name=data["name"],
                        args=data["arguments"],
                        id=f"call_{abs(hash(text))}" # 產生一個臨時 ID
                    )]
            except json.JSONDecodeError:
                pass
        return None

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成，支援 tools"""
        # 1. 檢查是否有綁定工具 (bind_tools 會將工具放入 kwargs)
        tools = kwargs.get("tools", None)
        
        # 2. 格式化 Prompt (包含工具定義)
        prompt = self._format_messages_to_prompt(messages, tools=tools)
        input_ids = torch.tensor(self._tokenizer.encode(prompt)).to(self._model.device)
        input_ids = input_ids.unsqueeze(0)

        # 3. 執行推論
        final_text = ""
        for token_text in self._inference_core(input_ids):
            final_text += token_text
            if run_manager:
                run_manager.on_llm_new_token(token_text)
        
        # 4. 解析輸出 (如果是工具呼叫)
        msg_kwargs = {"content": final_text}
        if tools:
            tool_calls = self._parse_tool_call(final_text)
            if tool_calls:
                msg_kwargs["tool_calls"] = tool_calls
                # 如果判定為 tool call，content 可以為空或保留思考過程
                # msg_kwargs["content"] = "" 
        
        return ChatResult(generations=[ChatGeneration(message=AIMessage(**msg_kwargs))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成"""
        # Stream 模式較難即時解析 JSON Tool Call，通常直接輸出文字
        # 但我們可以做簡單的 buffer 檢查，這裡先維持純文字輸出
        tools = kwargs.get("tools", None)
        prompt = self._format_messages_to_prompt(messages, tools=tools)
        input_ids = torch.tensor(self._tokenizer.encode(prompt)).to(self._model.device)
        input_ids = input_ids.unsqueeze(0)

        for token_text in self._inference_core(input_ids):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token_text))
            if run_manager:
                run_manager.on_llm_new_token(token_text, chunk=chunk)
            yield chunk

# --- 測試範例 ---
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    # 定義一個簡單工具
    @tool
    def get_weather(location: str):
        """Get the current weather in a given location"""
        return "Sunny, 25C"

    # 初始化
    MODEL_PATH = "D:\\LLM\\gemma\\gemma3_4b"
    llm = TransformersModel(model_name=MODEL_PATH)
    
    # 綁定工具
    llm_with_tools = llm.bind_tools([get_weather])
    
    # 測試
    print("--- Testing Tool Calling ---")
    query = "新竹現在的天氣如何？"
    result = llm_with_tools.invoke([HumanMessage(content=query)])
    
    print(f"User Query: {query}")
    print(f"Model Output Raw: {result.content}")
    if result.tool_calls:
        print(f"Detected Tool Call: {result.tool_calls}")
    else:
        print("No tool call detected.")

import os
import asyncio
import concurrent
import gc
import hashlib
import json
import logging
import uuid
import warnings
from datetime import datetime
from typing import Any, Dict

import pytz
from pydantic import ValidationError

from mem0.configs.base import MemoryConfig, MemoryItem
from mem0.configs.enums import MemoryType
from mem0.configs.prompts import (
    PROCEDURAL_MEMORY_SYSTEM_PROMPT,
    get_update_memory_messages,
)
from mem0.memory.base import MemoryBase
from mem0.memory.setup import setup_config, mem0_dir
from mem0.memory.storage import SQLiteManager
from mem0.memory.telemetry import capture_event
from mem0.memory.utils import (
    get_fact_retrieval_messages,
    parse_messages,
    parse_vision_messages,
    remove_code_blocks,
)
from mem0.utils.factory import EmbedderFactory, LlmFactory, VectorStoreFactory

# 设置用户配置
# Setup user config
setup_config()

logger = logging.getLogger(__name__)


class Memory(MemoryBase):
    """记忆系统的主类，实现了记忆的添加、检索、更新和删除等功能"""
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        """初始化记忆系统
        
        Args:
            config: 记忆系统配置，默认使用MemoryConfig的默认配置
        """
        self.config = config

        # 自定义提示词配置
        self.custom_fact_extraction_prompt = self.config.custom_fact_extraction_prompt
        self.custom_update_memory_prompt = self.config.custom_update_memory_prompt
        
        # 初始化嵌入模型、向量存储和LLM
        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )

        # 创建向量存储
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, 
            self.config.vector_store.config
        )
    
        # 创建LLM
        self.llm = LlmFactory.create(
            self.config.llm.provider, 
            self.config.llm.config
        )
        
        # 初始化数据库和集合名称
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        # 图形记忆功能默认关闭
        self.enable_graph = False

        # 如果配置了图形存储，则启用图形记忆功能
        if self.config.graph_store.config:
            from mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        self.config.vector_store.config.collection_name = "mem0_migrations"
        if self.config.vector_store.provider in ["faiss", "qdrant"]:
            provider_path = f"migrations_{self.config.vector_store.provider}"
            self.config.vector_store.config.path = os.path.join(mem0_dir, provider_path)
            os.makedirs(self.config.vector_store.config.path, exist_ok=True)

        self._telemetry_vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )

        capture_event("mem0.init", self, {"sync_type": "sync"})

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]):
        """从配置字典创建Memory实例
        
        Args:
            config_dict: 配置字典
            
        Returns:
            Memory实例
        """
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置字典，确保配置的完整性
        
        Args:
            config_dict: 配置字典
            
        Returns:
            处理后的配置字典
        """
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        memory_type=None,
        prompt=None,
    ):
        """
        创建新的记忆

        Args:
            messages (str or List[Dict[str, str]]): 要存储在记忆中的消息
            user_id (str, optional): 创建记忆的用户ID
            agent_id (str, optional): 创建记忆的代理ID
            run_id (str, optional): 创建记忆的运行ID
            metadata (dict, optional): 要与记忆一起存储的元数据
            filters (dict, optional): 应用于搜索的过滤器
            infer (bool, optional): 是否推断记忆，默认为True
            memory_type (str, optional): 要创建的记忆类型，默认为None。默认情况下，它创建短期记忆和长期（语义和情景）记忆。传递"procedural_memory"以创建程序记忆
            prompt (str, optional): 用于记忆创建的提示
            
        Returns:
            dict: 包含记忆添加操作结果的字典
            result: 受影响事件的字典，每个字典具有以下键:
              'memories': 受影响的记忆
              'graph': 受影响的图形记忆

              'memories'和'graph'是字典，每个都有以下子键:
                'add': 添加的记忆
                'update': 更新的记忆
                'delete': 删除的记忆
        """
        if metadata is None:
            metadata = {}

        filters = filters or {} # 过滤下面的三个信息
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        # 确保至少提供了一个过滤条件
        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        # 验证记忆类型
        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )
            
        # 消息格式化，将字符串转换为标准消息格式
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        # 如果是程序记忆类型，调用专门的创建方法
        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = self._create_procedural_memory(messages, metadata=metadata, prompt=prompt)
            return results
            
        # 系统检查是否需要处理视觉信息
        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)
            
        # 并行执行向量存储和图形记忆操作
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)    # 向量存储添加操作
            future2 = executor.submit(self._add_to_graph, messages, filters)    # 图形记忆添加流程

            concurrent.futures.wait([future1, future2])

            vector_store_result = future1.result()
            graph_result = future2.result()

        # 处理API版本兼容性
        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        # 返回结果，包括向量存储和图形记忆的结果
        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    def _add_to_vector_store(self, messages, metadata, filters, infer):
        """向向量存储添加记忆
        
        Args:
            messages: 消息列表
            metadata: 元数据
            filters: 过滤条件
            infer: 是否推断记忆
            
        Returns:
            添加的记忆列表
        """
        # 如果不需要推断，直接将消息添加到向量存储
        if not infer:
            returned_memories = []
            for message in messages:
                if message["role"] != "system":
                    message_embeddings = self.embedding_model.embed(message["content"], "add")
                    memory_id = self._create_memory(message["content"], message_embeddings, metadata)
                    returned_memories.append({"id": memory_id, "memory": message["content"], "event": "ADD"})
            return returned_memories

        # 解析消息
        parsed_messages = parse_messages(messages)

        # 准备提示词
        if self.config.custom_fact_extraction_prompt:   # 准备的提示prompt
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        # 事实提取，调用LLM生成响应
        response = self.llm.generate_response(  
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        # 解析LLM响应，提取事实
        try:
            response = remove_code_blocks(response) # 解析响应
            # 示例响应格式：
            # {
            #     "facts": [
            #         "用户喜欢吃苹果",
            #         "用户不喜欢吃香蕉"
            #     ]
            # }
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []
            
        # 为每个事实生成嵌入向量并搜索相似记忆
        retrieved_old_memory = []
        new_message_embeddings = {}
        for new_mem in new_retrieved_facts:
            # 生成嵌入向量
            messages_embeddings = self.embedding_model.embed(new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            # 搜索相似记忆
            existing_memories = self.vector_store.search(
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            for mem in existing_memories:
                retrieved_old_memory.append({"id": mem.id, "text": mem.payload["data"]})
            # 假设系统找到了一条相似记忆：{"id": "abc123", "text": "用户喜欢水果"}
            
        # 去重，确保每个记忆ID只出现一次
        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")
        
        # 映射UUID（防止LLM幻觉）
        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)
        # 处理后的记忆 [{"id": "0", "text": "用户喜欢水果"}]
        # UUID映射： {"0": "abc123"}

        # 使用LLM决定如何处理这些记忆
        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
        )

        # 调用LLM获取记忆操作决策
        try:
            new_memories_with_actions = self.llm.generate_response(
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
            new_memories_with_actions = []

        # 解析LLM响应
        try:
            new_memories_with_actions = remove_code_blocks(new_memories_with_actions)
            new_memories_with_actions = json.loads(new_memories_with_actions)
        except Exception as e:
            logging.error(f"Invalid JSON response: {e}")
            new_memories_with_actions = []
        # 假设LLM返回以下决策：
        # {
        #   "memory": [
        #     {
        #       "id": "0",
        #       "text": "用户喜欢水果，特别是苹果，但不喜欢香蕉",
        #       "event": "UPDATE",
        #       "old_memory": "用户喜欢水果"
        #     },
        #     {
        #       "text": "用户不喜欢吃香蕉",
        #       "event": "ADD"
        #     }
        #   ]
        # }
    
        # 执行记忆操作（添加、更新、删除）
        returned_memories = []
        try:
            for resp in new_memories_with_actions.get("memory", []):
                logging.info(resp)
                try:
                    if not resp.get("text"):
                        logging.info("Skipping memory entry because of empty `text` field.")
                        continue
                    elif resp.get("event") == "ADD":
                        # 创建新记忆
                        memory_id = self._create_memory(    
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": memory_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "UPDATE":
                        # 更新现有记忆
                        self._update_memory(    
                            memory_id=temp_uuid_mapping[resp["id"]],
                            data=resp.get("text"),
                            existing_embeddings=new_message_embeddings,
                            metadata=metadata,
                        )
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif resp.get("event") == "DELETE":
                        # 删除记忆
                        self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")])
                        returned_memories.append(
                            {
                                "id": temp_uuid_mapping[resp.get("id")],
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif resp.get("event") == "NONE":
                        logging.info("NOOP for Memory.")
                except Exception as e:
                    logging.error(f"Error in new_memories_with_actions: {e}")
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
        # 该例子中，系统会：
        # 1. 更新ID为"abc123"的记忆为"用户喜欢水果，特别是苹果，但不喜欢香蕉"
        # 2. 添加一条新记忆"用户不喜欢吃香蕉"

        # 记录事件
        capture_event(
            "mem0.add",
            self,
            {"version": self.api_version, "keys": list(filters.keys()), "sync_type": "sync"},
        )

        return returned_memories

    def _add_to_graph(self, messages, filters):
        """向图存储添加记忆
        
        Args:
            messages: 消息列表
            filters: 过滤条件
            
        Returns:
            添加的实体列表
        """
        # 图形记忆会提取实体和关系，例如：
        # - 实体： 用户 、 苹果 、 香蕉
        # - 关系： 用户-喜欢-苹果 、 用户-不喜欢-香蕉
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            # 提取非系统消息的内容
            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = self.graph.add(data, filters)

        return added_entities

    def get(self, memory_id):
        """
        通过ID检索记忆

        Args:
            memory_id (str): 要检索的记忆的ID

        Returns:
            dict: 检索到的记忆
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "sync"})
        memory = self.vector_store.get(vector_id=memory_id)
        if not memory:
            return None

        # 提取过滤条件
        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

        # 准备基本记忆项
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # 添加元数据（如果有额外的键）
        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        # 合并结果
        result = {**memory_item, **filters}

        return result

    def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        列出所有记忆

        Returns:
            list: 所有记忆的列表
        """
        # 构建过滤条件
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        capture_event("mem0.get_all", self, {"limit": limit, "keys": list(filters.keys()), "sync_type": "sync"})

        # 并行获取向量存储和图存储的记忆
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._get_all_from_vector_store, filters, limit)
            future_graph_entities = executor.submit(self.graph.get_all, filters, limit) if self.enable_graph else None

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            all_memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # 处理返回结果
        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}

        # 处理API版本兼容性
        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories
        else:
            return {"results": all_memories}

    def _get_all_from_vector_store(self, filters, limit):
        """从向量存储获取所有记忆
        
        Args:
            filters: 过滤条件
            limit: 限制数量
            
        Returns:
            记忆列表
        """
        # 从向量存储获取记忆
        memories = self.vector_store.list(filters=filters, limit=limit)

        # 定义要排除的键
        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        
        # 格式化记忆列表
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        搜索记忆

        Args:
            query (str): 要搜索的查询
            user_id (str, optional): 要搜索的用户ID
            agent_id (str, optional): 要搜索的代理ID
            run_id (str, optional): 要搜索的运行ID
            limit (int, optional): 限制结果数量
            filters (dict, optional): 应用于搜索的过滤器

        Returns:
            list: 搜索结果列表
        """
        # 构建过滤条件
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        # 确保至少提供了一个过滤条件
        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        # 记录事件
        capture_event(
            "mem0.search",
            self,
            {"limit": limit, "version": self.api_version, "keys": list(filters.keys()), "sync_type": "sync"},
        )

        # 并行搜索向量存储和图存储
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_memories = executor.submit(self._search_vector_store, query, filters, limit)
            future_graph_entities = executor.submit(self.graph.search, query, filters) if self.enable_graph else None

            concurrent.futures.wait(
                [future_memories, future_graph_entities] if future_graph_entities else [future_memories]
            )

            memories = future_memories.result()
            graph_entities = future_graph_entities.result() if future_graph_entities else None

        # 处理返回结果
        if self.enable_graph:
            return {"results": memories, "relations": graph_entities}

        # 处理API版本兼容性
        if self.api_version == "v1.0":
            warnings.warn(
                "The current search API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return memories
        else:
            return {"results": memories}

    def _search_vector_store(self, query, filters, limit):
        embeddings = self.embedding_model.embed(query, "search")
        memories = self.vector_store.search(query=query, vectors=embeddings, limit=limit, filters=filters)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    def update(self, memory_id, data):
        """
        Update a memory by ID.

        Args:
            query: 查询字符串
            filters: 过滤条件
            limit: 限制数量
            
        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "sync"})

        existing_embeddings = {data: self.embedding_model.embed(data, "update")}

        self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    def delete(self, memory_id):
        """
        Delete a memory by ID.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "sync"})
        self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        capture_event("mem0.delete_all", self, {"keys": list(filters.keys()), "sync_type": "sync"})
        memories = self.vector_store.list(filters=filters)[0]
        for memory in memories:
            self._delete_memory(memory.id)

        logger.info(f"Deleted {len(memories)} memories")

        if self.enable_graph:
            self.graph.delete_all(filters)

        return {"message": "Memories deleted successfully!"}

    def history(self, memory_id):
        """
        Get the history of changes for a memory by ID.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "sync"})
        return self.db.get_history(memory_id)

    def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        # 获取或生成嵌入向量
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, memory_action="add")
        memory_id = str(uuid.uuid4())   # # 生成唯一ID和元数据
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        # # 插入向量存储
        self.vector_store.insert(
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )
        # 记录历史变更
        self.db.add_history(memory_id, None, data, "ADD", created_at=metadata["created_at"])
        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _create_procedural_memory(self, messages, metadata=None, prompt=None):
        """
        Create a procedural memory

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {
                "role": "user",
                "content": "Create procedural memory of the above conversation.",
            },
        ]

        try:
            procedural_memory = self.llm.generate_response(messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        # Generate embeddings for the summary
        embeddings = self.embedding_model.embed(procedural_memory, memory_action="add")
        # Create the memory
        memory_id = self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "sync"})

        # Return results in the same format as add()
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:    # 获取现有的记忆
            existing_memory = self.vector_store.get(vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")
        prev_value = existing_memory.payload.get("data")
        # 准备新元数据
        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()
        # 保留原始过滤条件
        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]
        # 获取或生成嵌入向量
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = self.embedding_model.embed(data, "update")
        # 更新向量存储
        self.vector_store.update(
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )
        logger.info(f"Updating memory with ID {memory_id=} with {data=}")
        # 记录历史变更
        self.db.add_history(
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
        )
        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = self.vector_store.get(vector_id=memory_id)
        prev_value = existing_memory.payload["data"]
        self.vector_store.delete(vector_id=memory_id)
        self.db.add_history(memory_id, prev_value, None, "DELETE", is_deleted=1)
        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "sync"})
        return memory_id

    def reset(self):
        """
        Reset the memory store by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        self.vector_store.delete_col()

        gc.collect()

        # Close the client if it has a close method
        if hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'close'):
            self.vector_store.client.close()

        # Close the old connection if possible
        if hasattr(self.db, 'connection') and self.db.connection:
                self.db.connection.execute("DROP TABLE IF EXISTS history")
                self.db.connection.close()

        self.db = SQLiteManager(self.config.history_db_path)

        # Create a new vector store with the same configuration
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("mem0.reset", self, {"sync_type": "sync"})

    def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")


class AsyncMemory(MemoryBase):
    def __init__(self, config: MemoryConfig = MemoryConfig()):
        self.config = config

        self.embedding_model = EmbedderFactory.create(
            self.config.embedder.provider,
            self.config.embedder.config,
            self.config.vector_store.config,
        )
        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        self.llm = LlmFactory.create(self.config.llm.provider, self.config.llm.config)
        self.db = SQLiteManager(self.config.history_db_path)
        self.collection_name = self.config.vector_store.config.collection_name
        self.api_version = self.config.version

        self.enable_graph = False

        if self.config.graph_store.config:
            from mem0.memory.graph_memory import MemoryGraph

            self.graph = MemoryGraph(self.config)
            self.enable_graph = True

        capture_event("mem0.init", self, {"sync_type": "async"})

    @classmethod
    async def from_config(cls, config_dict: Dict[str, Any]):
        try:
            config = cls._process_config(config_dict)
            config = MemoryConfig(**config_dict)
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise
        return cls(config)

    @staticmethod
    def _process_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if "graph_store" in config_dict:
            if "vector_store" not in config_dict and "embedder" in config_dict:
                config_dict["vector_store"] = {}
                config_dict["vector_store"]["config"] = {}
                config_dict["vector_store"]["config"]["embedding_model_dims"] = config_dict["embedder"]["config"][
                    "embedding_dims"
                ]
        try:
            return config_dict
        except ValidationError as e:
            logger.error(f"Configuration validation error: {e}")
            raise

    async def add(
        self,
        messages,
        user_id=None,
        agent_id=None,
        run_id=None,
        metadata=None,
        filters=None,
        infer=True,
        memory_type=None,
        prompt=None,
        llm=None,
    ):
        """
        Create a new memory asynchronously.

        Args:
            messages (str or List[Dict[str, str]]): Messages to store in the memory.
            user_id (str, optional): ID of the user creating the memory. Defaults to None.
            agent_id (str, optional): ID of the agent creating the memory. Defaults to None.
            run_id (str, optional): ID of the run creating the memory. Defaults to None.
            metadata (dict, optional): Metadata to store with the memory. Defaults to None.
            filters (dict, optional): Filters to apply to the search. Defaults to None.
            infer (bool, optional): Whether to infer the memories. Defaults to True.
            memory_type (str, optional): Type of memory to create. Defaults to None. By default, it creates the short term memories and long term (semantic and episodic) memories. Pass "procedural_memory" to create procedural memories.
            prompt (str, optional): Prompt to use for the memory creation. Defaults to None.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
        Returns:
            dict: A dictionary containing the result of the memory addition operation.
            result: dict of affected events with each dict has the following key:
              'memories': affected memories
              'graph': affected graph memories

              'memories' and 'graph' is a dict, each with following subkeys:
                'add': added memory
                'update': updated memory
                'delete': deleted memory
        """
        if metadata is None:
            metadata = {}

        filters = filters or {}
        if user_id:
            filters["user_id"] = metadata["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = metadata["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = metadata["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        if memory_type is not None and memory_type != MemoryType.PROCEDURAL.value:
            raise ValueError(
                f"Invalid 'memory_type'. Please pass {MemoryType.PROCEDURAL.value} to create procedural memories."
            )

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        if agent_id is not None and memory_type == MemoryType.PROCEDURAL.value:
            results = await self._create_procedural_memory(messages, metadata=metadata, llm=llm, prompt=prompt)
            return results

        if self.config.llm.config.get("enable_vision"):
            messages = parse_vision_messages(messages, self.llm, self.config.llm.config.get("vision_details"))
        else:
            messages = parse_vision_messages(messages)

        # Run vector store and graph operations concurrently
        vector_store_task = asyncio.create_task(self._add_to_vector_store(messages, metadata, filters, infer))
        graph_task = asyncio.create_task(self._add_to_graph(messages, filters))

        vector_store_result, graph_result = await asyncio.gather(vector_store_task, graph_task)

        if self.api_version == "v1.0":
            warnings.warn(
                "The current add API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return vector_store_result

        if self.enable_graph:
            return {
                "results": vector_store_result,
                "relations": graph_result,
            }

        return {"results": vector_store_result}

    async def _add_to_vector_store(self, messages, metadata, filters, infer):
        if not infer:
            returned_memories = []
            for message in messages:
                if message["role"] != "system":
                    message_embeddings = await asyncio.to_thread(self.embedding_model.embed, message["content"], "add")
                    memory_id = await self._create_memory(message["content"], message_embeddings, metadata)
                    returned_memories.append({"id": memory_id, "memory": message["content"], "event": "ADD"})
            return returned_memories

        parsed_messages = parse_messages(messages)

        if self.config.custom_fact_extraction_prompt:
            system_prompt = self.config.custom_fact_extraction_prompt
            user_prompt = f"Input:\n{parsed_messages}"
        else:
            system_prompt, user_prompt = get_fact_retrieval_messages(parsed_messages)

        response = await asyncio.to_thread(
            self.llm.generate_response,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )

        try:
            response = remove_code_blocks(response)
            new_retrieved_facts = json.loads(response)["facts"]
        except Exception as e:
            logging.error(f"Error in new_retrieved_facts: {e}")
            new_retrieved_facts = []

        retrieved_old_memory = []
        new_message_embeddings = {}

        # Process all facts concurrently
        async def process_fact(new_mem):
            messages_embeddings = await asyncio.to_thread(self.embedding_model.embed, new_mem, "add")
            new_message_embeddings[new_mem] = messages_embeddings
            existing_memories = await asyncio.to_thread(
                self.vector_store.search,
                query=new_mem,
                vectors=messages_embeddings,
                limit=5,
                filters=filters,
            )
            return [(mem.id, mem.payload["data"]) for mem in existing_memories]

        fact_tasks = [process_fact(fact) for fact in new_retrieved_facts]
        fact_results = await asyncio.gather(*fact_tasks)

        # Flatten results and build retrieved_old_memory
        for result in fact_results:
            for mem_id, mem_data in result:
                retrieved_old_memory.append({"id": mem_id, "text": mem_data})

        unique_data = {}
        for item in retrieved_old_memory:
            unique_data[item["id"]] = item
        retrieved_old_memory = list(unique_data.values())
        logging.info(f"Total existing memories: {len(retrieved_old_memory)}")

        # mapping UUIDs with integers for handling UUID hallucinations
        temp_uuid_mapping = {}
        for idx, item in enumerate(retrieved_old_memory):
            temp_uuid_mapping[str(idx)] = item["id"]
            retrieved_old_memory[idx]["id"] = str(idx)

        function_calling_prompt = get_update_memory_messages(
            retrieved_old_memory, new_retrieved_facts, self.config.custom_update_memory_prompt
        )

        try:
            new_memories_with_actions = await asyncio.to_thread(
                self.llm.generate_response,
                messages=[{"role": "user", "content": function_calling_prompt}],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")
            new_memories_with_actions = []

        try:
            new_memories_with_actions = remove_code_blocks(new_memories_with_actions)
            new_memories_with_actions = json.loads(new_memories_with_actions)
        except Exception as e:
            logging.error(f"Invalid JSON response: {e}")
            new_memories_with_actions = []

        returned_memories = []
        try:
            memory_tasks = []
            for resp in new_memories_with_actions.get("memory", []):
                logging.info(resp)
                try:
                    if not resp.get("text"):
                        logging.info("Skipping memory entry because of empty `text` field.")
                        continue
                    elif resp.get("event") == "ADD":
                        task = asyncio.create_task(
                            self._create_memory(
                                data=resp.get("text"), existing_embeddings=new_message_embeddings, metadata=metadata
                            )
                        )
                        memory_tasks.append((task, resp, "ADD", None))
                    elif resp.get("event") == "UPDATE":
                        task = asyncio.create_task(
                            self._update_memory(
                                memory_id=temp_uuid_mapping[resp["id"]],
                                data=resp.get("text"),
                                existing_embeddings=new_message_embeddings,
                                metadata=metadata,
                            )
                        )
                        memory_tasks.append((task, resp, "UPDATE", temp_uuid_mapping[resp["id"]]))
                    elif resp.get("event") == "DELETE":
                        task = asyncio.create_task(self._delete_memory(memory_id=temp_uuid_mapping[resp.get("id")]))
                        memory_tasks.append((task, resp, "DELETE", temp_uuid_mapping[resp["id"]]))
                    elif resp.get("event") == "NONE":
                        logging.info("NOOP for Memory.")
                except Exception as e:
                    logging.error(f"Error in new_memories_with_actions: {e}")

            # Wait for all memory operations to complete
            for task, resp, event_type, mem_id in memory_tasks:
                try:
                    result_id = await task
                    if event_type == "ADD":
                        returned_memories.append(
                            {
                                "id": result_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                    elif event_type == "UPDATE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                                "previous_memory": resp.get("old_memory"),
                            }
                        )
                    elif event_type == "DELETE":
                        returned_memories.append(
                            {
                                "id": mem_id,
                                "memory": resp.get("text"),
                                "event": resp.get("event"),
                            }
                        )
                except Exception as e:
                    logging.error(f"Error processing memory task: {e}")

        except Exception as e:
            logging.error(f"Error in new_memories_with_actions: {e}")

        capture_event("mem0.add", self, {"version": self.api_version, "keys": list(filters.keys()), "sync_type": "async"})

        return returned_memories

    async def _add_to_graph(self, messages, filters):
        added_entities = []
        if self.enable_graph:
            if filters.get("user_id") is None:
                filters["user_id"] = "user"

            data = "\n".join([msg["content"] for msg in messages if "content" in msg and msg["role"] != "system"])
            added_entities = await asyncio.to_thread(self.graph.add, data, filters)

        return added_entities

    async def get(self, memory_id):
        """
        Retrieve a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to retrieve.

        Returns:
            dict: Retrieved memory.
        """
        capture_event("mem0.get", self, {"memory_id": memory_id, "sync_type": "async"})
        memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        if not memory:
            return None

        filters = {key: memory.payload[key] for key in ["user_id", "agent_id", "run_id"] if memory.payload.get(key)}

        # Prepare base memory item
        memory_item = MemoryItem(
            id=memory.id,
            memory=memory.payload["data"],
            hash=memory.payload.get("hash"),
            created_at=memory.payload.get("created_at"),
            updated_at=memory.payload.get("updated_at"),
        ).model_dump(exclude={"score"})

        # Add metadata if there are additional keys
        excluded_keys = {"user_id", "agent_id", "run_id", "hash", "data", "created_at", "updated_at", "id"}
        additional_metadata = {k: v for k, v in memory.payload.items() if k not in excluded_keys}
        if additional_metadata:
            memory_item["metadata"] = additional_metadata

        result = {**memory_item, **filters}

        return result

    async def get_all(self, user_id=None, agent_id=None, run_id=None, limit=100):
        """
        List all memories asynchronously.

        Returns:
            list: List of all memories.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        capture_event("mem0.get_all", self, {"limit": limit, "keys": list(filters.keys()), "sync_type": "async"})

        # Run vector store and graph operations concurrently
        vector_store_task = asyncio.create_task(self._get_all_from_vector_store(filters, limit))

        if self.enable_graph:
            graph_task = asyncio.create_task(asyncio.to_thread(self.graph.get_all, filters, limit))
            all_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            all_memories = await vector_store_task
            graph_entities = None

        if self.enable_graph:
            return {"results": all_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return all_memories
        else:
            return {"results": all_memories}

    async def _get_all_from_vector_store(self, filters, limit):
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters, limit=limit)

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }
        all_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                ).model_dump(exclude={"score"}),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories[0]
        ]
        return all_memories

    async def search(self, query, user_id=None, agent_id=None, run_id=None, limit=100, filters=None):
        """
        Search for memories asynchronously.

        Args:
            query (str): Query to search for.
            user_id (str, optional): ID of the user to search for. Defaults to None.
            agent_id (str, optional): ID of the agent to search for. Defaults to None.
            run_id (str, optional): ID of the run to search for. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to 100.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: List of search results.
        """
        filters = filters or {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not any(key in filters for key in ("user_id", "agent_id", "run_id")):
            raise ValueError("One of the filters: user_id, agent_id or run_id is required!")

        capture_event(
            "mem0.search",
            self,
            {"limit": limit, "version": self.api_version, "keys": list(filters.keys()), "sync_type": "async"},
        )

        # Run vector store and graph operations concurrently
        vector_store_task = asyncio.create_task(self._search_vector_store(query, filters, limit))

        if self.enable_graph:
            graph_task = asyncio.create_task(asyncio.to_thread(self.graph.search, query, filters, limit))
            original_memories, graph_entities = await asyncio.gather(vector_store_task, graph_task)
        else:
            original_memories = await vector_store_task
            graph_entities = None

        if self.enable_graph:
            return {"results": original_memories, "relations": graph_entities}

        if self.api_version == "v1.0":
            warnings.warn(
                "The current get_all API output format is deprecated. "
                "To use the latest format, set `api_version='v1.1'`. "
                "The current format will be removed in mem0ai 1.1.0 and later versions.",
                category=DeprecationWarning,
                stacklevel=2,
            )
            return original_memories
        else:
            return {"results": original_memories}

    async def _search_vector_store(self, query, filters, limit):
        embeddings = await asyncio.to_thread(self.embedding_model.embed, query, "search")
        memories = await asyncio.to_thread(
            self.vector_store.search, query=query, vectors=embeddings, limit=limit, filters=filters
        )

        excluded_keys = {
            "user_id",
            "agent_id",
            "run_id",
            "hash",
            "data",
            "created_at",
            "updated_at",
            "id",
        }

        original_memories = [
            {
                **MemoryItem(
                    id=mem.id,
                    memory=mem.payload["data"],
                    hash=mem.payload.get("hash"),
                    created_at=mem.payload.get("created_at"),
                    updated_at=mem.payload.get("updated_at"),
                    score=mem.score,
                ).model_dump(),
                **{key: mem.payload[key] for key in ["user_id", "agent_id", "run_id"] if key in mem.payload},
                **(
                    {"metadata": {k: v for k, v in mem.payload.items() if k not in excluded_keys}}
                    if any(k for k in mem.payload if k not in excluded_keys)
                    else {}
                ),
            }
            for mem in memories
        ]

        return original_memories

    async def update(self, memory_id, data):
        """
        Update a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to update.
            data (dict): Data to update the memory with.

        Returns:
            dict: Updated memory.
        """
        capture_event("mem0.update", self, {"memory_id": memory_id, "sync_type": "async"})

        embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")
        existing_embeddings = {data: embeddings}

        await self._update_memory(memory_id, data, existing_embeddings)
        return {"message": "Memory updated successfully!"}

    async def delete(self, memory_id):
        """
        Delete a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to delete.
        """
        capture_event("mem0.delete", self, {"memory_id": memory_id, "sync_type": "async"})
        await self._delete_memory(memory_id)
        return {"message": "Memory deleted successfully!"}

    async def delete_all(self, user_id=None, agent_id=None, run_id=None):
        """
        Delete all memories asynchronously.

        Args:
            user_id (str, optional): ID of the user to delete memories for. Defaults to None.
            agent_id (str, optional): ID of the agent to delete memories for. Defaults to None.
            run_id (str, optional): ID of the run to delete memories for. Defaults to None.
        """
        filters = {}
        if user_id:
            filters["user_id"] = user_id
        if agent_id:
            filters["agent_id"] = agent_id
        if run_id:
            filters["run_id"] = run_id

        if not filters:
            raise ValueError(
                "At least one filter is required to delete all memories. If you want to delete all memories, use the `reset()` method."
            )

        capture_event("mem0.delete_all", self, {"keys": list(filters.keys()), "sync_type": "async"})
        memories = await asyncio.to_thread(self.vector_store.list, filters=filters)

        delete_tasks = []
        for memory in memories[0]:
            delete_tasks.append(self._delete_memory(memory.id))

        await asyncio.gather(*delete_tasks)

        logger.info(f"Deleted {len(memories[0])} memories")

        if self.enable_graph:
            await asyncio.to_thread(self.graph.delete_all, filters)

        return {"message": "Memories deleted successfully!"}

    async def history(self, memory_id):
        """
        Get the history of changes for a memory by ID asynchronously.

        Args:
            memory_id (str): ID of the memory to get history for.

        Returns:
            list: List of changes for the memory.
        """
        capture_event("mem0.history", self, {"memory_id": memory_id, "sync_type": "async"})
        return await asyncio.to_thread(self.db.get_history, memory_id)

    async def _create_memory(self, data, existing_embeddings, metadata=None):
        logging.debug(f"Creating memory with {data=}")
        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, memory_action="add")

        memory_id = str(uuid.uuid4())
        metadata = metadata or {}
        metadata["data"] = data
        metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        await asyncio.to_thread(
            self.vector_store.insert,
            vectors=[embeddings],
            ids=[memory_id],
            payloads=[metadata],
        )

        await asyncio.to_thread(self.db.add_history, memory_id, None, data, "ADD", created_at=metadata["created_at"])

        capture_event("mem0._create_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _create_procedural_memory(self, messages, metadata=None, llm=None, prompt=None):
        """
        Create a procedural memory asynchronously

        Args:
            messages (list): List of messages to create a procedural memory from.
            metadata (dict): Metadata to create a procedural memory from.
            llm (BaseChatModel, optional): LLM class to use for generating procedural memories. Defaults to None. Useful when user is using LangChain ChatModel.
            prompt (str, optional): Prompt to use for the procedural memory creation. Defaults to None.
        """
        try:
            from langchain_core.messages.utils import (
                convert_to_messages,  # type: ignore
            )
        except Exception:
            logger.error(
                "Import error while loading langchain-core. Please install 'langchain-core' to use procedural memory."
            )
            raise

        logger.info("Creating procedural memory")

        parsed_messages = [
            {"role": "system", "content": prompt or PROCEDURAL_MEMORY_SYSTEM_PROMPT},
            *messages,
            {"role": "user", "content": "Create procedural memory of the above conversation."},
        ]

        try:
            if llm is not None:
                parsed_messages = convert_to_messages(parsed_messages)
                response = await asyncio.to_thread(llm.invoke, input=parsed_messages)
                procedural_memory = response.content
            else:
                procedural_memory = await asyncio.to_thread(self.llm.generate_response, messages=parsed_messages)
        except Exception as e:
            logger.error(f"Error generating procedural memory summary: {e}")
            raise

        if metadata is None:
            raise ValueError("Metadata cannot be done for procedural memory.")

        metadata["memory_type"] = MemoryType.PROCEDURAL.value
        # Generate embeddings for the summary
        embeddings = await asyncio.to_thread(self.embedding_model.embed, procedural_memory, memory_action="add")
        # Create the memory
        memory_id = await self._create_memory(procedural_memory, {procedural_memory: embeddings}, metadata=metadata)
        capture_event("mem0._create_procedural_memory", self, {"memory_id": memory_id, "sync_type": "async"})

        # Return results in the same format as add()
        result = {"results": [{"id": memory_id, "memory": procedural_memory, "event": "ADD"}]}

        return result

    async def _update_memory(self, memory_id, data, existing_embeddings, metadata=None):
        logger.info(f"Updating memory with {data=}")

        try:
            existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        except Exception:
            raise ValueError(f"Error getting memory with ID {memory_id}. Please provide a valid 'memory_id'")

        prev_value = existing_memory.payload.get("data")

        new_metadata = metadata or {}
        new_metadata["data"] = data
        new_metadata["hash"] = hashlib.md5(data.encode()).hexdigest()
        new_metadata["created_at"] = existing_memory.payload.get("created_at")
        new_metadata["updated_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()

        if "user_id" in existing_memory.payload:
            new_metadata["user_id"] = existing_memory.payload["user_id"]
        if "agent_id" in existing_memory.payload:
            new_metadata["agent_id"] = existing_memory.payload["agent_id"]
        if "run_id" in existing_memory.payload:
            new_metadata["run_id"] = existing_memory.payload["run_id"]

        if data in existing_embeddings:
            embeddings = existing_embeddings[data]
        else:
            embeddings = await asyncio.to_thread(self.embedding_model.embed, data, "update")

        await asyncio.to_thread(
            self.vector_store.update,
            vector_id=memory_id,
            vector=embeddings,
            payload=new_metadata,
        )

        logger.info(f"Updating memory with ID {memory_id=} with {data=}")

        await asyncio.to_thread(
            self.db.add_history,
            memory_id,
            prev_value,
            data,
            "UPDATE",
            created_at=new_metadata["created_at"],
            updated_at=new_metadata["updated_at"],
        )

        capture_event("mem0._update_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def _delete_memory(self, memory_id):
        logging.info(f"Deleting memory with {memory_id=}")
        existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=memory_id)
        prev_value = existing_memory.payload["data"]

        await asyncio.to_thread(self.vector_store.delete, vector_id=memory_id)
        await asyncio.to_thread(self.db.add_history, memory_id, prev_value, None, "DELETE", is_deleted=1)

        capture_event("mem0._delete_memory", self, {"memory_id": memory_id, "sync_type": "async"})
        return memory_id

    async def reset(self):
        """
        Reset the memory store asynchronously by:
            Deletes the vector store collection
            Resets the database
            Recreates the vector store with a new client
        """
        logger.warning("Resetting all memories")
        await asyncio.to_thread(self.vector_store.delete_col)

        gc.collect()

        if hasattr(self.vector_store, 'client') and hasattr(self.vector_store.client, 'close'):
            await asyncio.to_thread(self.vector_store.client.close)

        if hasattr(self.db, 'connection') and self.db.connection:
            await asyncio.to_thread(lambda: self.db.connection.execute("DROP TABLE IF EXISTS history"))
            await asyncio.to_thread(self.db.connection.close)

        self.db = SQLiteManager(self.config.history_db_path)

        self.vector_store = VectorStoreFactory.create(
            self.config.vector_store.provider, self.config.vector_store.config
        )
        capture_event("mem0.reset", self, {"sync_type": "async"})

    async def chat(self, query):
        raise NotImplementedError("Chat function not implemented yet.")

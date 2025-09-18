import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessageType(Enum):
    """메시지 타입 정의"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"


class AgentStatus(Enum):
    """에이전트 상태 정의"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Message:
    """A2A 프로토콜 메시지 구조"""
    id: str
    sender: str
    receiver: str
    type: MessageType
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """메시지를 딕셔너리로 변환"""
        return {
            **asdict(self),
            'timestamp': self.timestamp.isoformat(),
            'type': self.type.value
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """딕셔너리에서 메시지 생성"""
        return cls(
            id=data['id'],
            sender=data['sender'],
            receiver=data['receiver'],
            type=MessageType(data['type']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id')
        )


class BaseAgent(ABC):
    """모든 AI 에이전트의 베이스 클래스"""

    def __init__(self, agent_id: str, name: str):
        self.agent_id = agent_id
        self.name = name
        self.status = AgentStatus.IDLE
        self.message_queue = asyncio.Queue()
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.running = False
        self.message_history: List[Message] = []

        logger.info(f"Agent '{self.name}' ({self.agent_id}) initialized")

    async def start(self):
        """에이전트 시작"""
        self.running = True
        self.status = AgentStatus.IDLE
        logger.info(f"Agent '{self.name}' started")
        asyncio.create_task(self._message_processing_loop())

    async def stop(self):
        """에이전트 중지"""
        self.running = False
        self.status = AgentStatus.OFFLINE
        logger.info(f"Agent '{self.name}' stopped")

    async def _message_processing_loop(self):
        """메시지 처리 루프"""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                self.message_history.append(message)
                await self._handle_message(message)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message in {self.name}: {e}")
                self.status = AgentStatus.ERROR

    async def _handle_message(self, message: Message):
        """메시지 타입별 처리"""
        logger.info(f"{self.name} received {message.type.value} from {message.sender}")

        try:
            if message.type == MessageType.REQUEST:
                await self._handle_request(message)
            elif message.type == MessageType.RESPONSE:
                await self._handle_response(message)
            elif message.type == MessageType.NOTIFICATION:
                await self._handle_notification(message)
            elif message.type == MessageType.ERROR:
                await self._handle_error(message)

        except Exception as e:
            logger.error(f"Error handling {message.type.value}: {e}")
            await self._send_error_response(message, str(e))

    async def _handle_request(self, message: Message):
        """요청 메시지 처리"""
        self.status = AgentStatus.BUSY

        try:
            response_content = await self.process_request(message.content)

            response = Message(
                id=str(uuid.uuid4()),
                sender=self.agent_id,
                receiver=message.sender,
                type=MessageType.RESPONSE,
                content=response_content,
                timestamp=datetime.now(),
                correlation_id=message.id
            )

            await self.send_message(response)

        except Exception as e:
            await self._send_error_response(message, str(e))
        finally:
            self.status = AgentStatus.IDLE

    async def _handle_response(self, message: Message):
        """응답 메시지 처리"""
        correlation_id = message.correlation_id
        if correlation_id and correlation_id in self.response_handlers:
            future = self.response_handlers.pop(correlation_id)
            future.set_result(message.content)

    async def _handle_notification(self, message: Message):
        """알림 메시지 처리"""
        await self.process_notification(message.content)

    async def _handle_error(self, message: Message):
        """에러 메시지 처리"""
        logger.error(f"Error from {message.sender}: {message.content}")
        await self.process_error(message.content)

    async def _send_error_response(self, original_message: Message, error_msg: str):
        """에러 응답 전송"""
        error_response = Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=original_message.sender,
            type=MessageType.ERROR,
            content={"error": error_msg, "original_message_id": original_message.id},
            timestamp=datetime.now(),
            correlation_id=original_message.id
        )
        await self.send_message(error_response)

    async def send_message(self, message: Message):
        """다른 에이전트에게 메시지 전송"""
        logger.info(f"{self.name} sending {message.type.value} to {message.receiver}")

    async def send_request(self, receiver: str, content: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """요청 전송 및 응답 대기"""
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            sender=self.agent_id,
            receiver=receiver,
            type=MessageType.REQUEST,
            content=content,
            timestamp=datetime.now()
        )

        future = asyncio.Future()
        self.response_handlers[message_id] = future

        try:
            await self.send_message(message)
            response = await asyncio.wait_for(future, timeout=timeout)
            return response
        except asyncio.TimeoutError:
            self.response_handlers.pop(message_id, None)
            raise Exception(f"Request timeout to {receiver}")

    async def send_notification(self, receiver: str, content: Dict[str, Any]):
        """알림 전송"""
        message = Message(
            id=str(uuid.uuid4()),
            sender=self.agent_id,
            receiver=receiver,
            type=MessageType.NOTIFICATION,
            content=content,
            timestamp=datetime.now()
        )
        await self.send_message(message)

    def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보 반환"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "message_queue_size": self.message_queue.qsize(),
            "message_history_count": len(self.message_history),
            "pending_responses": len(self.response_handlers)
        }

    @abstractmethod
    async def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """요청 처리 (각 에이전트가 구현)"""
        pass

    async def process_notification(self, content: Dict[str, Any]):
        """알림 처리 (선택적 구현)"""
        pass

    async def process_error(self, content: Dict[str, Any]):
        """에러 처리 (선택적 구현)"""
        pass


if __name__ == "__main__":
    class TestAgent(BaseAgent):
        async def process_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": f"Processed: {content}"}

    async def main():
        agent = TestAgent("test-001", "테스트 에이전트")
        await agent.start()
        print("Base Agent test completed!")
        await agent.stop()

    asyncio.run(main())
import asyncio
import logging
from collections.abc import AsyncGenerator

from a2a.server.agent_execution import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    Part,
    Task,
    TaskState,
    TaskStatus,
    UnsupportedOperationError,
)
from google.adk import Runner
from google.adk.events import Event
from google.genai import types


logger = logging.getLogger(__name__)


class WareHouseManagerAgentExecutor(AgentExecutor):
    def __init__(self, runner: Runner):
        self.init(runner)

    def init(self, runner: Runner):
        self.runner = runner
        self._running_sessions: dict[str, asyncio.Task[None]] = {}

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        A2A -> ADK -> A2A execution flow.

        A2A RequestContext
            |
            v
        Extract A2A message
            |
            v
        Convert to ADK Content
            |
            v
        Runner.run_async()
            |
            v
        Stream ADK Events
            |
            v
        Convert ADK Events to A2A artifacts
            |
            v
        Mark A2A task completed
        """

        task: Task | None = context.current_task

        # ---------------------------------------------------------
        # Create the A2A task if this is the first request.
        # A2A task lifecycle streams must start with a Task.
        # ---------------------------------------------------------

        if task is None:
            task = Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_SUBMITTED,
                ),
            )

            await event_queue.enqueue_event(task)

        task_id = task.id
        context_id = task.context_id

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        # ---------------------------------------------------------
        # Tell the A2A client that execution has started.
        # ---------------------------------------------------------

        await updater.update_status(
            TaskState.TASK_STATE_WORKING,
        )

        # ---------------------------------------------------------
        # Run ADK as a separate asyncio task.
        #
        # Keeping a reference allows cancel() to cancel the
        # underlying ADK execution.
        # ---------------------------------------------------------

        adk_task = asyncio.create_task(
            self._run_adk(
                context=context,
                updater=updater,
            )
        )

        self._running_sessions[task_id] = adk_task

        try:
            await adk_task

            # -----------------------------------------------------
            # ADK has finished.
            #
            # All response chunks have already been sent as
            # TaskArtifactUpdateEvents.
            #
            # This is the final A2A lifecycle event.
            # -----------------------------------------------------

            await updater.complete()

        except asyncio.CancelledError:
            logger.info(
                "Warehouse manager task %s was cancelled",
                task_id,
            )

            await updater.cancel()

            raise

        except Exception as exc:
            logger.exception(
                "Warehouse manager task %s failed",
                task_id,
            )

            await updater.failed(
                message=self._create_error_message(str(exc)),
            )

            raise

        finally:
            self._running_sessions.pop(task_id, None)

    async def _run_adk(
        self,
        context: RequestContext,
        updater: TaskUpdater,
    ) -> None:
        """
        Convert the A2A request into an ADK request and stream
        ADK events back into A2A artifacts.
        """

        # ---------------------------------------------------------
        # A2A -> ADK
        # ---------------------------------------------------------

        text = self._get_message_text(context)

        if not text:
            raise ValueError(
                "A2A request does not contain a text message."
            )

        user_id = self._get_user_id(context)
        session_id = self._get_session_id(context)

        adk_message = types.Content(
            role="user",
            parts=[
                types.Part(
                    text=text,
                ),
            ],
        )

        # ---------------------------------------------------------
        # Ensure the ADK session exists before streaming.
        # ---------------------------------------------------------

        existing_session = await self.runner.session_service.get_session(
            app_name=self.runner.app_name,
            user_id=user_id,
            session_id=session_id,
        )
        if existing_session is None:
            await self.runner.session_service.create_session(
                app_name=self.runner.app_name,
                user_id=user_id,
                session_id=session_id,
            )

        # ---------------------------------------------------------
        # ADK streaming
        # ---------------------------------------------------------

        artifact_created = False
        async for event in self._run_adk_stream(
            user_id=user_id,
            session_id=session_id,
            message=adk_message,
        ):
            artifact_created = await self._handle_adk_event(
                event=event,
                updater=updater,
                artifact_created=artifact_created,
            )

    async def _run_adk_stream(
        self,
        user_id: str,
        session_id: str,
        message: types.Content,
    ) -> AsyncGenerator[Event, None]:
        """
        Run the ADK Runner and expose its events as an async
        generator.

        This preserves ADK's streaming behavior.
        """

        async for event in self.runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            yield event

    async def _handle_adk_event(
        self,
        event: Event,
        updater: TaskUpdater,
        artifact_created: bool = False,
    ) -> bool:
        """
        Convert an ADK Event into an A2A artifact update.

        Every textual ADK event is streamed to the A2A client.
        """

        if event.content is None:
            return artifact_created

        if not event.content.parts:
            return artifact_created

        for adk_part in event.content.parts:
            if not adk_part.text:
                continue

            # -----------------------------------------------------
            # ADK Part -> A2A Part
            #
            # A2A v1 uses Part(text=...) directly.
            # -----------------------------------------------------

            a2a_part = Part(
                text=adk_part.text,
            )

            # -----------------------------------------------------
            # Stream the chunk to the A2A client.
            #
            # append=True means the client can reconstruct the
            # response from multiple chunks.
            # -----------------------------------------------------

            await updater.add_artifact(
                parts=[a2a_part],
                name="response",
                append=artifact_created,
            )
            artifact_created = True

        return artifact_created

    def _get_message_text(
        self,
        context: RequestContext,
    ) -> str:
        """
        Convert the incoming A2A Message into plain text.
        """

        message = context.message

        if message is None:
            return ""

        text_parts: list[str] = []

        for part in message.parts:
            if part.text:
                text_parts.append(part.text)

        return "".join(text_parts)

    def _get_user_id(
        self,
        context: RequestContext,
    ) -> str:
        """
        Map the A2A caller to an ADK user.

        Replace this with your authenticated user ID.
        """

        return "a2a-user"

    def _get_session_id(
        self,
        context: RequestContext,
    ) -> str:
        """
        Map the A2A conversation to an ADK session.

        Using context_id means multiple A2A messages in the same
        conversation use the same ADK session.
        """

        return context.context_id

    def _create_error_message(
        self,
        error: str,
    ) -> types.Content:
        """
        Create an ADK Content object representing an error.

        This method also intentionally uses the google.genai
        types import for consistency with the ADK side of the
        adapter.
        """

        return types.Content(
            role="model",
            parts=[
                types.Part(
                    text=f"Agent execution failed: {error}",
                ),
            ],
        )

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Cancel the currently running ADK execution for an A2A task.
        """

        task_id = context.task_id

        if not task_id:
            raise UnsupportedOperationError(
                message="No task ID was provided.",
            )

        running_task = self._running_sessions.get(task_id)

        if running_task is None:
            raise UnsupportedOperationError(
                message=f"Task {task_id} is not currently running.",
            )

        logger.info(
            "Cancelling warehouse manager task %s",
            task_id,
        )

        running_task.cancel()

        try:
            await running_task
        except asyncio.CancelledError:
            pass
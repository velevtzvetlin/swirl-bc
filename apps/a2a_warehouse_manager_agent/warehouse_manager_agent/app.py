import logging
import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    AgentInterface
)

from agent import WarehouseManagerAgent
from agent_executor import WareHouseManagerAgentExecutor
from dotenv import load_dotenv
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from starlette.applications import Starlette
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.agent_card_routes import create_agent_card_routes

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HOST = "localhost"
PORT = 10001

def main():
    # Initialize the agent
    capabilities = AgentCapabilities(
        streaming = True
    )
    skill_availability = AgentSkill(
        id="ABC",
        name="Check Warehouse Availability",
        description="Check availability of items across warehouses.",
        tags=["availability", "warehouse", "warehouses"],
        examples=["What is the availability of the item 123?"],
    )
    skill_reservation = AgentSkill(
        id="DEF",
        name="Reserve Warehouse Items",
        description="Reserve items in the warehouses.",
        tags=["reservation", "warehouse", "warehouses"],
        examples=["Reserve 5 items of items of the item 123 from the closest warehouse to the user based on Vilnius, Lithuania location."],
    )
    agent_card = AgentCard(
        name="Warehouse Manager Agent",
        description="The user is asking to reserve items from the warehouses or about availability of items in the warehouses.",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=[skill_availability, skill_reservation],
        capabilities=capabilities,
        supported_interfaces=[AgentInterface(
            protocol_binding="JSONRPC",
            protocol_version="1.0",
            url=f"http://{HOST}:{PORT}/"
        )]
    )
    adk_agent = WarehouseManagerAgent().get_agent() # this is the exposing of the agent
    runner = Runner(
        agent=adk_agent,
        app_name=agent_card.name,
        session_service=InMemorySessionService(),
        artifact_service=InMemoryArtifactService(),
        memory_service=InMemoryMemoryService(),
    )

    agent_executor = WareHouseManagerAgentExecutor(runner=runner)
    request_handler = DefaultRequestHandler(
        agent_card=agent_card, 
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    app = Starlette(routes=[
        *create_jsonrpc_routes(request_handler, rpc_url="/"),
        *create_agent_card_routes(agent_card)
    ])
    
    uvicorn.run(app, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
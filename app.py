import os
import json
import asyncio
import logging
import re
import importlib.util
from typing import Any, Dict, List, Callable, Optional, Pattern, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import instructor
from openai import AsyncOpenAI
import subprocess

# Load environment variables from .env file
load_dotenv()

class TaskType(str, Enum):
    Task = "Task"
    Choice = "Choice"
    Parallel = "Parallel"
    Map = "Map"
    Wait = "Wait"

class AgentaTask(BaseModel):
    Type: TaskType
    Resource: str
    Parameters: Dict[str, Any] = Field(default_factory=dict)
    Next: Optional[str] = None
    End: Optional[bool] = None
    Catch: Optional[List[Dict[str, Any]]] = None
    Retry: Optional[List[Dict[str, Any]]] = None
    ResultPath: Optional[str] = None
    InputPath: Optional[str] = None
    OutputPath: Optional[str] = None

class AgentaStateMachine(BaseModel):
    StartAt: str
    States: Dict[str, AgentaTask]

class AgenticApp:
    class Task(BaseModel):
        task: str
        is_atomic: bool
        parameters: Dict[str, Any] = Field(default_factory=dict)
        output_schema: Optional[Dict[str, Any]] = None

    class TaskSchema(BaseModel):
        tasks: List['AgenticApp.Task']

    class Hook:
        def __init__(self, pattern: Optional[str], callback: Callable, match_type: str):
            self.pattern = pattern
            self.callback = callback
            self.match_type = match_type
            self.compiled_pattern: Optional[Pattern] = None
            if self.match_type == 'regex' and self.pattern:
                self.compiled_pattern = re.compile(self.pattern)

        def matches(self, event: Dict[str, Any]) -> bool:
            event_str = json.dumps(event)
            if self.match_type == 'regex' and self.compiled_pattern:
                return bool(self.compiled_pattern.search(event_str))
            elif self.match_type == 'exact':
                return self.pattern == event_str
            elif self.match_type == 'type':
                type_mapping = {
                    'str': str, 'int': int, 'float': float,
                    'bool': bool, 'dict': dict, 'list': list,
                }
                expected_type = type_mapping.get(self.pattern)
                if expected_type:
                    return any(isinstance(value, expected_type) for value in event.values())
            return False

    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "sk-1234567890abcdef1234567890abcdef")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4")
        self.base_url = "http://localhost:8080/v1"
        self.openai = instructor.patch(AsyncOpenAI(api_key=self.openai_api_key, base_url=self.base_url))
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                            handlers=[logging.StreamHandler()])
        self.logger = logging.getLogger("AgenticApp")

        self.extensions: Dict[str, Callable[..., Any]] = {}
        self.hooks: List['AgenticApp.Hook'] = []
        self.agents: Dict[str, 'AgenticApp.MetaAgent'] = {}
        self.processed_tasks: set = set()

        self.load_extensions()
        self.logger.info("AgenticApp initialized.")

    def load_extensions(self):
        extensions_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extensions')
        if not os.path.isdir(extensions_dir):
            self.logger.warning(f"Extensions directory '{extensions_dir}' does not exist.")
            return

        for filename in os.listdir(extensions_dir):
            file_path = os.path.join(extensions_dir, filename)
            if filename.endswith('.py'):
                module_name = filename[:-3]
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(module)
                        if hasattr(module, 'Extension'):
                            extension_class = getattr(module, 'Extension')
                            if hasattr(extension_class, 'execute') and callable(getattr(extension_class, 'execute')):
                                extension_instance = extension_class()
                                self.register_extension(extension_instance.name, extension_instance.execute)
                                self.logger.info(f"Loaded extension '{extension_instance.name}' from '{filename}'.")
                        elif hasattr(module, 'execute') and callable(getattr(module, 'execute')):
                            self.register_extension(module_name, module.execute)
                            self.logger.info(f"Loaded extension '{module_name}' from '{filename}'.")
                        else:
                            self.logger.warning(f"No valid extension found in '{filename}'.")
                    except Exception as e:
                        self.logger.error(f"Failed to load extension '{filename}': {e}")
            elif os.access(file_path, os.X_OK):
                extension_name = os.path.splitext(filename)[0]
                self.register_extension(extension_name, lambda **kwargs: self.run_binary_extension(file_path, kwargs))
                self.logger.info(f"Loaded binary extension '{extension_name}' from '{filename}'.")

    async def run_binary_extension(self, path: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        try:
            args = [path] + [f"--{key}" for key, value in parameters.items() for item in (key, str(value))]
            self.logger.info(f"Running binary extension '{path}' with args: {args}")
            proc = await asyncio.create_subprocess_exec(*args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Binary extension '{path}' failed with error: {stderr.decode().strip()}")
            return json.loads(stdout.decode())
        except Exception as e:
            self.logger.error(f"Error running binary extension '{path}': {e}")
            raise

    def register_extension(self, name: str, extension: Callable[..., Any]):
        if not callable(extension):
            raise ValueError("Extension must be callable.")
        self.extensions[name] = extension
        self.logger.info(f"Extension '{name}' registered.")

    def register_hook(self, pattern: Optional[str], callback: Callable, match_type: str = 'exact'):
        if match_type not in ['regex', 'exact', 'type']:
            raise ValueError("match_type must be 'regex', 'exact', or 'type'.")
        hook = self.Hook(pattern, callback, match_type)
        self.hooks.append(hook)
        self.logger.info(f"Hook registered with pattern '{pattern}' and match_type '{match_type}'.")

    def trigger_hooks(self, event: Dict[str, Any]):
        for hook in self.hooks:
            try:
                if hook.matches(event):
                    self.logger.debug(f"Triggering hook for event: {event}")
                    hook.callback(event)
            except Exception as e:
                self.logger.error(f"Error in hook callback: {e}")

    async def decompose_to_workflow(self, high_level_task: str) -> AgentaStateMachine:
        try:
            response = await self.openai.chat.completions.create(
                model=self.openai_model,
                response_model=AgentaStateMachine,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that decomposes tasks into AWS Step Functions compatible workflows."},
                    {"role": "user", "content": f"Create a workflow for the following high-level task, formatted as an AWS Step Functions state machine: {high_level_task}"},
                ],
            )
            return response
        except Exception as e:
            self.logger.error(f"Error in decompose_to_workflow: {str(e)}")
            raise

    async def execute_workflow(self, workflow: AgentaStateMachine) -> Dict[str, Any]:
        current_state = workflow.StartAt
        context = {}

        while True:
            task = workflow.States[current_state]
            
            try:
                if task.Type == TaskType.Task:
                    result = await self.execute_task(task, context)
                    context = self.update_context(context, result, task.ResultPath)
                elif task.Type == TaskType.Choice:
                    next_state = self.evaluate_choice(task, context)
                    if next_state:
                        current_state = next_state
                        continue
                elif task.Type == TaskType.Parallel:
                    results = await self.execute_parallel_tasks(task, context)
                    context = self.update_context(context, results, task.ResultPath)
                elif task.Type == TaskType.Map:
                    results = await self.execute_map_task(task, context)
                    context = self.update_context(context, results, task.ResultPath)
                elif task.Type == TaskType.Wait:
                    await self.execute_wait_task(task)
                
                if task.End:
                    break
                
                current_state = task.Next
            except Exception as e:
                self.logger.error(f"Error executing task {current_state}: {str(e)}")
                if task.Catch:
                    for catcher in task.Catch:
                        if isinstance(e, eval(catcher["ErrorEquals"])):
                            current_state = catcher["Next"]
                            break
                    else:
                        raise
                else:
                    raise

        return context

    async def execute_task(self, task: AgentaTask, context: Dict[str, Any]) -> Any:
        if task.Resource.startswith("arn:agenta:ai::function:"):
            function_name = task.Resource.split(":")[-1]
            return await self.execute_function(function_name, task.Parameters, context)
        else:
            sub_workflow = await self.decompose_to_workflow(task.Resource)
            return await self.execute_workflow(sub_workflow)

    async def execute_function(self, function_name: str, parameters: Dict[str, Any], context: Dict[str, Any]) -> Any:
        if function_name not in self.extensions:
            raise ValueError(f"Unknown function: {function_name}")
        return await self.extensions[function_name](**parameters, context=context)

    def update_context(self, context: Dict[str, Any], result: Any, result_path: Optional[str]) -> Dict[str, Any]:
        if result_path is None:
            return result
        elif result_path.startswith("$"):
            json_path = result_path[1:].split(".")
            current = context
            for key in json_path[:-1]:
                current = current.setdefault(key, {})
            current[json_path[-1]] = result
        return context

    def evaluate_choice(self, task: AgentaTask, context: Dict[str, Any]) -> Optional[str]:
        for choice in task.Parameters.get("Choices", []):
            if self.evaluate_condition(choice, context):
                return choice["Next"]
        return task.Parameters.get("Default")

    def evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        variable = self.get_json_path(context, condition["Variable"])
        operator = next(iter(set(condition.keys()) - {"Variable"}))
        value = condition[operator]
        
        if operator == "StringEquals":
            return variable == value
        elif operator == "NumericEquals":
            return float(variable) == float(value)
        
        return False

    def get_json_path(self, data: Dict[str, Any], path: str) -> Any:
        current = data
        for key in path.lstrip("$").split("."):
            if key in current:
                current = current[key]
            else:
                return None
        return current

    async def execute_parallel_tasks(self, task: AgentaTask, context: Dict[str, Any]) -> List[Any]:
        sub_tasks = task.Parameters.get("Branches", [])
        results = await asyncio.gather(*[self.execute_workflow(AgentaStateMachine(**sub_task)) for sub_task in sub_tasks])
        return results

    async def execute_map_task(self, task: AgentaTask, context: Dict[str, Any]) -> List[Any]:
        items = self.get_json_path(context, task.Parameters["ItemsPath"])
        sub_task = AgentaStateMachine(**task.Parameters["Iterator"])
        results = await asyncio.gather(*[self.execute_workflow(sub_task) for _ in items])
        return results

    async def execute_wait_task(self, task: AgentaTask):
        if "Seconds" in task.Parameters:
            await asyncio.sleep(task.Parameters["Seconds"])
        elif "Timestamp" in task.Parameters:
            pass  # Implement timestamp-based waiting if needed

    def generate_agent_id(self, task: str) -> str:
        sanitized_task = re.sub(r'\s+', '_', task.strip())[:10]
        return f"{sanitized_task}_{len(self.agents) + 1}"

    async def execute_high_level_task(self, high_level_task: str) -> List[Any]:
        self.logger.info(f"Executing high-level task: {high_level_task}")
        workflow = await self.decompose_to_workflow(high_level_task)
        results = await self.execute_workflow(workflow)
        self.logger.info("All tasks executed.")
        return [results]

    def subscribe_to_agent_events(self, agent_id: str, hook_pattern: Optional[str], callback: Callable, match_type: str = 'exact'):
        def wrapped_callback(event: Dict[str, Any]):
            if event.get("agent_id") == agent_id:
                callback(event)
        self.register_hook(pattern=hook_pattern, callback=wrapped_callback, match_type=match_type)
        self.logger.info(f"Subscribed to events from agent '{agent_id}' with pattern '{hook_pattern}' and match_type '{match_type}'.")

    class MetaAgent:
        def __init__(self, app: 'AgenticApp', task: 'AgenticApp.Task', agent_id: str):
            self.app = app
            self.task = task
            self.agent_id = agent_id
            self.logger = logging.getLogger(f"AgenticApp.MetaAgent.{self.agent_id}")
            self.logger.info(f"MetaAgent '{self.agent_id}' initialized for task: {task.task}")

        async def execute(async def execute(self):
            self.logger.info(f"Agent '{self.agent_id}' executing task: {self.task.task}")
            result = await self.app.execute_task(self.task, agent_id=self.agent_id)
            self.logger.info(f"Agent '{self.agent_id}' completed task: {self.task.task} with result: {result}")
            return result

    async def run(self, high_level_task: str):
        try:
            final_results = await self.execute_high_level_task(high_level_task)
            self.logger.info("Final Results:")
            print(json.dumps(final_results, indent=4))
        except Exception as e:
            self.logger.error(f"An error occurred during execution: {e}")

    class BookFlightsExtension:
        name = "Book Flights"

        async def execute(self, date: str, destination: str) -> Dict[str, Any]:
            logger = logging.getLogger("BookFlightsExtension")
            logger.info(f"Booking flights to {destination} on {date}.")
            await asyncio.sleep(2)  # Simulate API call delay
            return {
                "task": "Book Flights",
                "status": "completed",
                "details": {
                    "date": date,
                    "destination": destination,
                    "flight_number": "AB123",
                    "price": "$500"
                }
            }

    class CheckHotelAvailabilityExtension:
        name = "Check Hotel Availability"

        async def execute(self, date: str, location: str) -> Dict[str, Any]:
            logger = logging.getLogger("CheckHotelAvailabilityExtension")
            logger.info(f"Checking hotel availability in {location} on {date}.")
            await asyncio.sleep(2)  # Simulate API call delay
            return {
                "task": "Check Hotel Availability",
                "status": "completed",
                "details": {
                    "date": date,
                    "location": location,
                    "hotel": "Hotel XYZ",
                    "rooms_available": 5
                }
            }

    class FindFamousRestaurantsExtension:
        name = "Find Famous Restaurants"

        async def execute(self, location: str) -> Dict[str, Any]:
            logger = logging.getLogger("FindFamousRestaurantsExtension")
            logger.info(f"Finding famous restaurants in {location}.")
            await asyncio.sleep(2)  # Simulate API call delay
            return {
                "task": "Find Famous Restaurants",
                "status": "completed",
                "details": {
                    "location": location,
                    "restaurants": ["Restaurant A", "Restaurant B", "Restaurant C"]
                }
            }

    def setup_default_extensions(self):
        book_flights_ext = self.BookFlightsExtension()
        check_hotel_ext = self.CheckHotelAvailabilityExtension()
        find_restaurants_ext = self.FindFamousRestaurantsExtension()

        self.register_extension(book_flights_ext.name, book_flights_ext.execute)
        self.register_extension(check_hotel_ext.name, check_hotel_ext.execute)
        self.register_extension(find_restaurants_ext.name, find_restaurants_ext.execute)
        self.logger.info("Default extensions registered.")

if __name__ == "__main__":
    agent_app = AgenticApp()
    agent_app.setup_default_extensions()

    def on_task_start(event: Dict[str, Any]):
        task = event.get("task")
        agent_id = event.get("agent_id")
        print(f"[Hook] Agent '{agent_id}' started task: {task['task']}")

    def on_task_complete(event: Dict[str, Any]):
        task = event.get("task")
        result = event.get("result")
        agent_id = event.get("agent_id")
        print(f"[Hook] Agent '{agent_id}' completed task: {task['task']} with result: {result}")

    def on_task_fail(event: Dict[str, Any]):
        task = event.get("task")
        error = event.get("error")
        agent_id = event.get("agent_id")
        print(f"[Hook] Agent '{agent_id}' failed task: {task['task']} with error: {error}")

    agent_app.register_hook(
        pattern=r'"event": "task_start"',
        callback=on_task_start,
        match_type='regex'
    )
    agent_app.register_hook(
        pattern=r'"event": "task_complete"',
        callback=on_task_complete,
        match_type='regex'
    )
    agent_app.register_hook(
        pattern=r'"event": "task_fail"',
        callback=on_task_fail,
        match_type='regex'
    )

    high_level_task = "Plan a trip to Melbourne, find famous restaurants, book flights, and check hotel availability."

    asyncio.run(agent_app.run(high_level_task))

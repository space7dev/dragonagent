from __future__ import annotations

import ast
from calendar import c
import inspect
import json
import os
import random
import re
import subprocess
import sys
import sys
import textwrap
import time
import traceback
from enum import Enum
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
import logging

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

run_id=None

STOP_INSTRUCTION=textwrap.dedent("""
# 🎨 
DO NOT generate `observation:` in your response. It will be provided by user for you.
Generate only SINGLE triplet of `next_thought`, `next_tool_name`, `next_tool_args` in your response.
""")

DEFAULT_PROXY_URL = os.getenv("SANDBOX_PROXY_URL", "http://sandbox_proxy")
DEFAULT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", "2200"))

PROBLEM_TYPE_CREATE = "CREATE"
PROBLEM_TYPE_FIX = "FIX"

GLM_MODEL_NAME = "zai-org/GLM-4.6-FP8"
GLM_MODEL_NAME_P = "zai-org/GLM-4.5-FP8"
KIMI_MODEL_NAME = "moonshotai/Kimi-K2-Instruct"
DEEPSEEK_MODEL_NAME = "deepseek-ai/DeepSeek-V3-0324"
QWEN_MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"
AGENT_MODELS = [model for model in [GLM_MODEL_NAME, KIMI_MODEL_NAME, GLM_MODEL_NAME_P, DEEPSEEK_MODEL_NAME, QWEN_MODEL_NAME] for _ in range(2)]
MAX_FIX_TASK_STEPS = 400
LATEST_OBSERVATIONS_TO_KEEP = 10
MAX_TOOL_CALL_RESPONSE_TOKENS = 3000
SUMMARIZE_BATCH_SIZE = 5
ESTIMATED_INPUT_COST = 0
ESTIMATED_OUTPUT_COST = 0
TOTAL_COST_THRESHOLD = 1.8
PRICES_PER_MODEL = {
    GLM_MODEL_NAME: [1.9, 2.0],
    GLM_MODEL_NAME_P: [1.4, 2.0],
    DEEPSEEK_MODEL_NAME: [1.25, 2.0],
    QWEN_MODEL_NAME: [1.5, 2],
    KIMI_MODEL_NAME: [0.39, 1.9]
}

DO_NOT_REPEAT_TOOL_CALLS=textwrap.dedent("""
You're not allowed to repeat the same tool call with the same arguments.
Your previous response: 
{previous_response}

Try to use something different!
""")

FORMAT_PROMPT_V0=textwrap.dedent("""
**📝 Response Format Requirements**

1. **Strict Triplet Format**:
   - `next_thought`: Detailed reasoning (include:
     - Problem understanding
     - Code analysis
     - Solution justification
     - Validation plan)
   - `next_tool_name`: Must be an exact tool name from the tool list
   - `next_tool_args`: Valid JSON with:
     - Proper escaping
     - No trailing commas
     - Tool-specific parameters

2. **Error Handling Format**:
   - For errors: 
     next_thought: "Error: [detailed explanation]"
     next_tool_name: ""
     next_tool_args: {}

3. **Example Valid Format**:
   next_thought: "I'll fix the JSON parsing issue by adding proper error handling and validation"
   next_tool_name: "apply_code_edit"
   next_tool_args: {
     "file_path": "network.py",
     "search": "return json.loads(response)",
     "replace": "try:\n    return json.loads(response)\nexcept JSONDecodeError:\n    print(f'Invalid JSON: {{response}}')\n    raise"
   }

4. **Invalid Format Examples** (Avoid These):
   - Missing any of the three required fields
   - JSON syntax errors in next_tool_args
   - Extra text outside the triplet format
   - Using incorrect tool names
   - Not quoting special characters properly
""")

FIX_TASK_SYSTEM_PROMPT = textwrap.dedent("""
# Hey there! You're a Coding Assistant 🚀. I have uploaded all files of a python repository. Your current working directory is at the root of that repo. You will be provided with a problem statement and you need to make the necessary changes to fix the issue.

## Recommended: Use Systematic Debugging Tools
When debugging complex code:
- Use `trace_execution` to manually simulate code line-by-line and identify where behavior diverges
- Use `identify_boundary_conditions` to ensure your fix handles edge cases (empty, None, single element, max/min)
These tools prevent guessing and ensure complete fixes instead of partial ones.

## Follow these steps to fix the issue:
1. As a first step, find the relevant files in the repo to work on.
2. Localise the code causing the issue.
3. Edit the sourcecode of the repo to resolve the issue.
4. Think about edgecases and make sure the fix handles them as well.
4.5. For complex logic, use trace_execution to simulate the code with failing inputs before editing.
5. Code must always be backward compatible unless explicitly mentioned otherwise in the problem statement.
6. Thoroughly check the entire code base to ensure the changes made are exhaustive and does not break any other functionality.
7. Thoroughly check the entire code base to ensure the changes user requested are only limited to the ones you have identified.
8. Never edit/update the existing test files directly when validating a hypothesis. Instead, when you need a new or focused test to reproduce or protect the fix, use the dedicated test generation tool.
9. Do not create any new files or directories unless absolutely necessary for the fix. Generated tests are allowed but are excluded from the final patch automatically.
10. Always check all the test cases which will be impacted with your change and ensure they don't fail.
11. You need to propose at least 2 meaningfully different and accurate solutions to the problem to the user for approval.
12. You need to look at both expected output mentioned in the problem statement AND the output in the most relevant test case. This is very important.
13. If the finish tool raises an error, rewind the workflow to the state from three steps earlier and resume from there.
14. If you find that the error while running the run_code or run_repo_tests tool due to missing dependencies, do not try to solve it as you don't have any internet access.
15. You can add debug prints and then run_repo_tests. For debug prints: use `print("DEBUG: <message>")` or `print(f"DEBUG: <message> {{<variable>}}")`
16. **CRITICAL - Final verification**: After implementing your fix, you MUST review the problem statement again and verify that ALL issues, features, and requirements mentioned have been completely addressed. Check each point in the problem statement systematically. If any issue or feature remains unsolved or partially implemented, you must continue working until everything is fully resolved before calling the finish tool.

## Multi-file awareness (critical):
- Tests and patch contexts may span multiple files. Do not stop after the first similar match or applied fix.
- Keep searching the repository after each match and apply consistent changes to every relevant file before finishing.
- Prefer using `search_in_all_files_content` to enumerate matches across the codebase and `search_in_specified_file_v2` to drill into each file; iterate until no applicable occurrences remain.
- Re-run tests only after covering all discovered occurrences to avoid partial fixes.

You have access to the following tools:-
{tools_docs}

Here is the problem statement:
{problem_statement}

{format_prompt}""")


FIX_TASK_INSTANCE_PROMPT_TEMPLATE = textwrap.dedent("""
# Now let's start. Here is the problem statement:
{problem_statement}
""")


TEMPERATURE_DETERMINATION_SYSTEM_PROMPT = """
You are selecting the sampling temperature for generating a Python script that solves the problem below. Output a single temperature in [0.3, 0.9] plus a short justification.

Problem to solve
<paste the exact problem statement, constraints, grading method, and any style/stack rules>

How to decide (use this rubric):

0.3–0.5 (more deterministic, correctness-focused):
Auto-graded tests, single or standard algorithms, clear constraints, API contracts, security/compliance, or strong need for reproducibility. Exploration is allowed but should be modest; correctness over variety.

0.5–0.7 (moderate exploration):
Multiple valid solution approaches, some design or library choices, ambiguous heuristics, or performance/complexity trade-offs. You want a balance between reliability and diversity of reasonable implementations.

0.7–0.9 (highly exploratory/creative):
Under-specified or open-ended tasks (e.g., data pipelines, architecture design, unusual constraints), where diverse solution structures, patterns, or strategies are valuable before refinement.

Never exceed 0.9, and don’t go below 0.3. If in doubt and auto-grading is involved, bias toward the 0.3–0.5 range.

Adjustments:

Push toward 0.3–0.5 for: strict reproducibility, flaky or rate-limited APIs, security/safety concerns, exact schemas, or unit-test–driven evaluation.

Push toward 0.5–0.7 for: standard problems with meaningful design choices or several equally valid patterns.

Push toward 0.7–0.9 for: idea generation, multiple alternative designs, exploratory analysis, or when creativity and variety are explicitly desired.

If you intend to generate tests first or follow a strict plan, you may lower the temperature one notch within these bands.
Output format (JSON only):

{
"temperature": 0.00,
"reason": "one concise sentence citing the rubric category"
}   

Think briefly, apply the rubric, then produce only the JSON.
"""


VERSION_COMPATIBILITY_FIX = """
import sys, pytest, collections, collections.abc, urllib3.exceptions, _pytest.pytester, numpy;
collections.Mapping = collections.abc.Mapping;
collections.MutableMapping = collections.abc.MutableMapping;
collections.MutableSet = collections.abc.MutableSet;
collections.Sequence = collections.abc.Sequence;
collections.Callable = collections.abc.Callable;
collections.Iterable = collections.abc.Iterable;
collections.Iterator = collections.abc.Iterator;
urllib3.exceptions.SNIMissingWarning = urllib3.exceptions.DependencyWarning;
pytest.RemovedInPytest4Warning = DeprecationWarning;
_pytest.pytester.Testdir = _pytest.pytester.Pytester;
numpy.PINF = numpy.inf;
numpy.unicode_ = numpy.str_;
numpy.bytes_ = numpy.bytes_;
numpy.float_ = numpy.float64;
numpy.string_ = numpy.bytes_;
numpy.NaN = numpy.nan;
"""

PYTEST_COMMAND_TEMPLATE = textwrap.dedent("""\
python -c "{version_compatibility_fix}
sys.exit(pytest.main([{file_paths}, '-vv', '-s', '--tb=long', '-W', 'ignore']))"\
""")

class EnhancedCOT:
    class Action:          
        def __init__(self, next_thought: str, next_tool_name: str, next_tool_args: dict, observation: list|tuple|str,is_error:bool=False,raw_response:str=None,total_attempts:int=0,inference_error_counter:dict=None,request_data:list=None):
            self.next_thought=next_thought
            self.next_tool_name=next_tool_name
            self.next_tool_args=next_tool_args
            self.observation=";".join(observation) if isinstance(observation,list) else observation
            self.is_error=is_error
            self.raw_response=raw_response
            self.total_attempts=total_attempts
            self.inference_error_counter=inference_error_counter
            self.request_data=request_data
            self.is_deleted=False

    def __init__(self,latest_observations_to_keep=5, summarize_batch_size=10):
        self.thoughts: list[EnhancedCOT.Action] = []
        self.latest_observations_to_keep=latest_observations_to_keep
        self.repeated_thoughts = 0
        self.summarize_batch_size = summarize_batch_size
        # Store summaries: key is (start_idx, end_idx) tuple, value is summary string
        self.summaries: dict[tuple[int, int], str] = {}
        # Track which indices have been summarized
        self.summarized_ranges: list[tuple[int, int]] = []

    def add_action(self, action: EnhancedCOT.Action) -> bool: # don't add if thought is repeated
        self.thoughts.append(action)
        # Check if we need to summarize older messages
        # Only check when we have enough messages to potentially summarize
        total_thoughts = len(self.thoughts)
        if total_thoughts >= self.latest_observations_to_keep + self.summarize_batch_size:
            self._check_and_summarize_if_needed()
        return True
    
    def _check_and_summarize_if_needed(self):
        """Check if we need to summarize older messages and trigger summarization."""
        total_thoughts = len(self.thoughts)
        cutoff_idx = total_thoughts - self.latest_observations_to_keep
        
        if cutoff_idx < self.summarize_batch_size:
            return  # Not enough messages to summarize yet
        
        # Find the oldest unsummarized index
        # Start from 0 and find gaps in summarized ranges
        oldest_unsummarized = 0
        for start, end in sorted(self.summarized_ranges):
            if start <= oldest_unsummarized < end:
                oldest_unsummarized = end
            elif start > oldest_unsummarized:
                break  # Found a gap, use oldest_unsummarized
        
        # Only summarize if we have a batch ready and it's before the cutoff
        if oldest_unsummarized >= cutoff_idx:
            return  # All messages before cutoff are already summarized or being kept
        
        # Calculate the range to summarize (don't go beyond cutoff)
        summarize_start = oldest_unsummarized
        summarize_end = min(summarize_start + self.summarize_batch_size, cutoff_idx)
        
        # Only summarize if we have a full batch (or at least summarize_batch_size messages)
        # This ensures incomplete batches remain unsummarized
        batch_size = summarize_end - summarize_start
        if batch_size >= self.summarize_batch_size:
            # Check if this range is already summarized
            range_key = (summarize_start, summarize_end)
            if range_key not in self.summaries:
                summary = self._summarize_messages_batch(summarize_start, summarize_end)
                if summary:
                    self.summaries[range_key] = summary
                    self.summarized_ranges.append(range_key)
                    self.summarized_ranges.sort()
    
    def _summarize_messages_batch(self, start_idx: int, end_idx: int) -> Optional[str]:
        """Summarize a batch of messages using LLM."""
        if start_idx >= end_idx or end_idx > len(self.thoughts):
            return None
        
        # Build the conversation to summarize
        conversation_parts = []
        for i in range(start_idx, end_idx):
            thought = self.thoughts[i]
            if thought.is_deleted:
                continue
            
            # Format the thought and tool call
            assistant_part = f"next_thought: {thought.next_thought}\n"
            assistant_part += f"next_tool_name: {thought.next_tool_name}\n"
            assistant_part += f"next_tool_args: {thought.next_tool_args}\n"
            
            # Format observation (truncate very long observations for summarization)
            if isinstance(thought.observation, (list, tuple)):
                try:
                    obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                except Exception:
                    obs_render = str(thought.observation)
            else:
                obs_render = str(thought.observation) if thought.observation else ""
            
            # Truncate very long observations to avoid token limits during summarization
            if len(obs_render) > 40000:
                obs_render = obs_render[:40000] + "... [truncated for summarization]"
            
            user_part = f"observation: {obs_render}"
            
            conversation_parts.append({
                "assistant": assistant_part,
                "user": user_part,
                "is_error": thought.is_error
            })
        
        if not conversation_parts:
            return None
        
        # Build the prompt for summarization
        conversation_text = ""
        for i, part in enumerate(conversation_parts, 1):
            conversation_text += f"\n--- Step {i} ---\n"
            conversation_text += f"Assistant: {part['assistant']}\n"
            # Observation already truncated to 2000 chars, show more context (up to 1500) for summarization
            user_obs = part['user']
            if len(user_obs) > 40000:
                user_obs = user_obs[:40000] + "... [truncated]"
            conversation_text += f"User: {user_obs}\n"
            if part['is_error']:
                conversation_text += "[Error occurred]\n"
        
        summarization_prompt = textwrap.dedent(f"""
        You are summarizing a conversation history between an AI agent and its environment.
        
        Summarize the following conversation steps concisely, focusing on:
        1. Key actions taken (tools used, files modified, tests run)
        2. Important findings or errors encountered
        3. Progress made toward solving the problem
        4. Critical decisions or changes in approach
        
        Keep the summary concise (2-4 sentences per step) but preserve important details.
        
        Conversation to summarize:
        {conversation_text}
        
        Provide a concise summary:
        """)
        
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that summarizes conversation history concisely."},
                {"role": "user", "content": summarization_prompt}
            ]
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.0)
            return response.strip()
        except Exception as e:
            return None
    
    def _get_summary_for_index(self, idx: int) -> Optional[str]:
        """Get the summary for a given message index if it exists."""
        for (start, end), summary in self.summaries.items():
            if start <= idx < end:
                return summary
        return None
        
    def is_thought_repeated(self)->bool:
        if len(self.thoughts) < 2:
            self.repeated_thoughts = 0
            return False
        last = self.thoughts[-1]
        prev = self.thoughts[-2]
        if last.next_tool_name == prev.next_tool_name and last.next_tool_args == prev.next_tool_args:
            self.repeated_thoughts += 1
            return True
        self.repeated_thoughts = 0
        return False

    def to_str(self):
        messages=[]
        # Track if we've added a summary for the current batch
        last_summary_range = None
        
        for i,thought in enumerate(self.thoughts):
            if thought.is_deleted:
                continue
            
            if i<len(self.thoughts)-self.latest_observations_to_keep:
                # Check if this index is part of a summarized range
                summary = self._get_summary_for_index(i)
                
                if summary:
                    # Find the range this summary covers
                    found_range = False
                    for (start, end), summ in self.summaries.items():
                        if start <= i < end:
                            # Only add summary once per range
                            if (start, end) != last_summary_range:
                                messages.append({
                                    "role": "system",
                                    "content": f"[Summarized conversation history (steps {start+1} to {end}):\n{summary}\n]"
                                })
                                last_summary_range = (start, end)
                            found_range = True
                            break  # Found the range, break out of inner loop
                    
                    # Skip individual messages in this range - continue outer loop
                    if found_range:
                        continue
                
                # If no summary available, show the message as-is (full content)
                assistant_str = f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                # Render list observations as JSON array for the model
                if isinstance(thought.observation, (list, tuple)):
                    try:
                        obs_render = json.dumps(list(thought.observation), ensure_ascii=False)
                    except Exception:
                        obs_render = str(thought.observation)
                else:
                    obs_render = str(thought.observation) if thought.observation else ""
                user_str = f"observation: {obs_render}"
                messages.append({"role":"assistant","content":assistant_str})
                messages.append({"role":"user","content":user_str})
                
            else:
                # Latest observations - always show full content
                if thought.is_error is None or i==len(self.thoughts)-1:
                    assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                    # Render list observations as JSON array for the model
                    if isinstance(thought.observation, (list, tuple)):
                        try:
                            obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                        except Exception:
                            obs_render=str(thought.observation)
                    else:
                        obs_render=str(thought.observation)
                    user_str=f"observation: {obs_render}"
                else:
                    if self.thoughts[-1].is_error==None and thought.is_error!=None:
                        assistant_str = (
                            f"next_thought:{thought.next_thought}\n"
                            f"next_tool_name:{thought.next_tool_name}\n"
                            f"next_tool_args:{thought.next_tool_args}")
                        if thought.observation is None:
                            _obs_len = 0
                        elif isinstance(thought.observation, (list, tuple)):
                            _obs_len = len(thought.observation)
                        else:
                            _obs_len = len(str(thought.observation).splitlines())
                        user_str=(
                            f"observation: error ocurred. detailed output omitted "
                            f"({_obs_len}) lines\n"
                        )
                    else:
                        assistant_str=f"next_thought:{thought.next_thought}\nnext_tool_name:{thought.next_tool_name}\nnext_tool_args:{thought.next_tool_args}"
                        if isinstance(thought.observation, (list, tuple)):
                            try:
                                obs_render=json.dumps(list(thought.observation), ensure_ascii=False)
                            except Exception:
                                obs_render=str(thought.observation)
                        else:
                            obs_render=str(thought.observation)
                        user_str=f"observation: {obs_render}"
                messages.append({"role":"assistant","content":assistant_str})
                messages.append({"role":"user","content":user_str})
        return messages

class EnhancedToolManager:
    logs = []
    PATCH_FILE_EXTENSIONS = (".py", ".ini", ".cfg", ".toml")

    class Error(Exception):
        class ErrorType(Enum):
            SYNTAX_ERROR=1
            RUNTIME_ERROR=2
            TIMEOUT=3
            FILE_NOT_FOUND=4
            SEARCH_TERM_NOT_FOUND=5
            UNKNOWN=6
            THIRD_PARTY_DEPENDENCIES=7
            MULTIPLE_SEARCH_RESULTS_FOUND=8
            BUG_REPORT_REQUIRED=9
            INVALID_RESPONSE_FORMAT=10
            INVALID_TOOL_NAME=11
            INVALID_FILE_PATH=12
            INVALID_TOOL_CALL=13
            IMPORT_ERROR=14
            
        def __init__(self,error_type:ErrorType,message:str):    
            self.error_type=error_type
            self.message=message

    def tool(fn):
        def wrapper(self, *args, **kwargs):
            self.tool_invocations[fn.__name__]+=1
            try:
                return fn(self, *args, **kwargs)
            except EnhancedToolManager.Error as e:
                self.tool_failure[fn.__name__][e.error_type]+=1
                return e.message

        # Preserve original function metadata
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        wrapper.__signature__ = inspect.signature(fn)
        wrapper.__annotations__ = fn.__annotations__.copy()
        wrapper.is_tool=True

        return wrapper
    
    @classmethod
    def tool_parsing(cls,fn):
        tool_schemas = None
        name = fn.__name__
        doc_fn = fn.__doc__ or ""
        # remove parameters section from here to be put in args section
        doc=doc_fn.split("Arguments:")[0]
        output_description=doc_fn.split("Output:")
        if len(output_description)>1:
            output_description="Output: "+output_description[1].strip()
            doc=doc+"\n\n"+output_description
        sig = inspect.signature(fn)
        properties = {}
        required = []
        for param in sig.parameters.values():
            if param.name == 'self':
                continue
            if param.default is param.empty and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY):
                required.append(param.name)
            type_hint = str(param.annotation) if param.annotation != param.empty else "string"
            param_description=re.search(f"{param.name}:([^\n]+)",doc_fn)
            if param_description:
                param_description=param_description.group(1)
            else:
                raise ValueError(f"Parameter description not found for {param.name} in {doc_fn}: tool name: {name}")
            # Special handling for list[str] / List[str] annotations so that the
            # generated JSON schema correctly represents an array of strings.
            if ("list" in type_hint.lower()) and ("str" in type_hint):
                properties[param.name] = {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": param_description
                }
                continue
            elif 'str' in type_hint:
                json_type = "string"
            elif 'int' in type_hint:
                json_type = "integer"
            elif 'float' in type_hint:
                json_type = "number"
            elif 'bool' in type_hint:
                json_type = "boolean"
            else:
                json_type = "string"
            properties[param.name] = {
                "type": json_type,
                "description": param_description
            }
        parameters = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        tool_schemas={
            "name": name,
            "description": doc.strip(),
            "input_schema": parameters
        }
        
        return tool_schemas

    @classmethod
    def get_tool_args_for_tool(cls,tool_name:str,required_only:bool=False)->list[str]:
        if tool_name not in cls.TOOL_LIST:
            return []  # Return empty list instead of error string to match return type
        if not required_only: 
            return list(cls.TOOL_LIST[tool_name]['input_schema']['properties'].keys())
        else:
            return cls.TOOL_LIST[tool_name]['input_schema']['required']
    
    @classmethod
    def _check_dependency_errors(self, output: str, extra_signatures: list[str] = []) -> bool:
        text = output.lower()
        return any(sig.lower() in text for sig in ["ModuleNotFoundError", "No module named", "ImportError", "pkg_resources.DistributionNotFound", "pkg_resources.VersionConflict", "INTERNALERROR", "Could not find a version that satisfies the requirement", "No matching distribution found for", "not configured", "missing module named", "missing dependency", "Failed to import", "Could not import", "cannot import", "cannot open shared object file", "undefined symbol", "bad magic number", "incompatible library"] + extra_signatures)


    def get_tool_docs(self)->str:
        return '\n\n'.join([json.dumps(tool_metadata, ensure_ascii=False) for _,tool_metadata in self.TOOL_LIST.items()])

    def get_tool(self,tool_name:str):
        if tool_name not in self.TOOL_LIST:
            return f"Error: tool '{tool_name}' not found"
        tool_method = getattr(self, tool_name, None)
        if tool_method is None or not callable(tool_method):
            return f"Error: tool '{tool_name}' does not exist. Please use one of the following tools: {', '.join(self.TOOL_LIST.keys())}"
        
        return tool_method

    def has_pending_patch_changes(self) -> bool:
        try:
            staged = subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            ).stdout.splitlines()
        except Exception as e:

            staged = []

        if any(self._is_patch_candidate_path(path) for path in staged):
            return True

        unstaged = self._collect_unstaged_patch_files()
        return bool(unstaged)

    def get_final_git_patch(self) -> str:
        try:
            to_add = self._collect_unstaged_patch_files()
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )
            patch_text = diff.stdout or ""
            return patch_text
        except Exception as e:
            return f"Error generating git patch: {e}"
    
    def _patch_exclude_paths(self) -> set[str]:
        exclude = {"src/agent.py", "src/agent_runner.py"}
        try:
            for _p in getattr(self, "generated_test_files", []):
                exclude.add(os.path.relpath(_p))
        except Exception:
            pass
        return exclude

    def _is_patch_candidate_path(self, path: str) -> bool:
        candidate = path.strip()
        if not candidate:
            return False
        if not candidate.endswith(self.PATCH_FILE_EXTENSIONS):
            return False
        return candidate not in self._patch_exclude_paths()

    def _collect_unstaged_patch_files(self) -> list[str]:
        try:
            result = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except Exception as e:

            return []

        files: list[str] = []
        for line in result.stdout.splitlines():
            path = line.strip()
            if self._is_patch_candidate_path(path):
                files.append(path)
        return files
            
class EnhancedNetwork:
    class ErrorType(Enum):
        EMPTY_RESPONSE=1
        RESERVED_TOKEN_PRESENT=2
        RATE_LIMIT_EXCEEDED=3
        INVALID_RESPONSE_FORMAT=4
        TIMEOUT=5
        UNKNOWN=6
        NETWORK_ERROR=7
        AUTHENTICATION_ERROR=8
        RESOURCE_EXHAUSTED=9
    
    @classmethod
    def is_valid_response(cls,raw_text:str)->bool:
        if type(raw_text) is dict and raw_text.get("error",None) is not None and raw_text.get("error")!="":
            return False,cls.ErrorType.EMPTY_RESPONSE.name
        
        # Check if response is complete - should contain all three required fields
        stripped = raw_text.strip()
        has_next_thought = "next_thought" in raw_text.lower() or "<next_thought>" in raw_text.lower()
        has_next_tool_name = "next_tool_name" in raw_text.lower() or "<next_tool_name>" in raw_text.lower()
        has_next_tool_args = "next_tool_args" in raw_text.lower() or "<next_tool_args>" in raw_text.lower()
        
        # Valid endings: JSON format or XML-style tags
        valid_ending = (stripped.endswith("}") or stripped.endswith("}]") or 
                       stripped.endswith("</next_tool_args>") or stripped.endswith(">"))
        
        # If has all fields but doesn't end properly, it's likely truncated
        if len(raw_text)==0:
            return False, cls.ErrorType.EMPTY_RESPONSE.name
        if has_next_thought and has_next_tool_name and has_next_tool_args and not valid_ending:
            return False, "Incomplete response, your response must be shorter to fit within context limit"
        if "<|reserved_token_" in raw_text:
            return False, cls.ErrorType.RESERVED_TOKEN_PRESENT.name
        if 'API request failed with status 429' in raw_text:
            return False, cls.ErrorType.RATE_LIMIT_EXCEEDED.name
        if 'Read timed out' in raw_text:
            return False, cls.ErrorType.TIMEOUT.name
        if 'Network unreachable' in raw_text or 'Connection refused' in raw_text:
            return False, cls.ErrorType.NETWORK_ERROR.name
        return True, None

    @classmethod
    def get_error_counter(cls)->dict[str,int]:
        return {
            k:0 for k in cls.ErrorType.__members__
        }   

    @classmethod
    def fix_json_string_with_llm(cls,json_string:str,attempt:int=0)->dict:
        logger.debug(f"Attempting to fix JSON string with LLM (attempt {attempt})")
        messages=[
            {"role":"system", "content":"Fix the json string sent by the user.  Reply only with the json string and nothing else."},
            {"role":"user", "content":json_string}
        ]
        response=cls.make_request(messages, model=DEEPSEEK_MODEL_NAME)
        try:
            response=response.replace('```json','').strip('```')
            response=json.loads(response)
            logger.debug("Successfully fixed JSON string with LLM")
            return response
        except JSONDecodeError as e:
            logger.warning(f"Failed to fix JSON string with LLM: {e}")
            return None
    
    @classmethod
    def make_request(cls,messages:list,model:str,attempt:int=0, temperature:float=0.0)->str:
        global run_id, ESTIMATED_INPUT_COST, ESTIMATED_OUTPUT_COST
        url = f"{DEFAULT_PROXY_URL.rstrip('/')}/api/inference"

        # Cache miss - make the actual request
        request_data = {
                "run_id": run_id if run_id else str(uuid4()),
                "messages": messages,
                "temperature": temperature,
            }

        headers = {
            "Content-Type": "application/json"
        }
        request_data['model'] = model
        input_tokens = Utils.count_tokens(messages)
        ESTIMATED_INPUT_COST += input_tokens * PRICES_PER_MODEL[model][0] / 1e6
        try:
            response = requests.post(url, json=request_data, timeout=120, headers=headers)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after 120 seconds for model {model}")
            return f"ERROR: Request timeout for model {model}"
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for model {model}: {e}")
            return f"ERROR: Connection failed for model {model}"
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for model {model}: {e}")
            logger.error(f"Response content: {response.json().get('detail','')}...")
            return f"ERROR: HTTP error {e.response.status_code} for model {model}"
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for model {model}: {e}")
            return f"ERROR: Request failed for model {model}"
        
        try:
            response_json = response.json()
        except JSONDecodeError as e:
            logger.error(f"Invalid JSON response for model {model}: {e}")
            logger.error(f"Response content: {response.text[:500]}...")
            return f"ERROR: Invalid JSON response for model {model}"
        
        try:
            is_oai_interface= type(response_json) is dict and response_json.get('choices') is not None and len(response_json.get('choices'))>0 and response_json.get('choices')[0].get('message') is not None
            if is_oai_interface:
                raw_text=response_json['choices'][0]['message']['content']
            else:
                if type(response_json) is str:
                    raw_text=response_json.strip("\n").strip()
                else:
                    raw_text=response_json
            if type(raw_text) is not dict:
                raw_text=raw_text.lstrip()

            output_tokens = Utils.count_tokens(raw_text)
            ESTIMATED_OUTPUT_COST += output_tokens * PRICES_PER_MODEL[model][1] / 1e6

            logger.info(f"[REQUEST] run_id: {run_id}, model: {model}, Estimated input tokens: {input_tokens}, Estimated output tokens: {output_tokens}")
            return raw_text
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error parsing response structure for model {model}: {e}")
            logger.error(f"Response JSON: {response_json}")
            return f"ERROR: Invalid response structure for model {model}"
        except Exception as e:
            logger.error(f"Unexpected error processing response for model {model}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"ERROR: Unexpected error for model {model}"

    @classmethod
    def _request_next_action_with_retry(cls, messages: dict, model: str, max_retries: int = 10, base_delay: float = 1.0, temperature: float = 0.0) -> str:
        raw_text='not defined'
        error_counter=cls.get_error_counter()
        next_thought, next_tool_name, next_tool_args = None, None, None
        total_attempts=0
        logger.info(f"Starting inference with model {model}, max_retries={max_retries}, temperature={temperature}")
        for attempt in range(max_retries):
            try:
                total_attempts+=1
                index = AGENT_MODELS.index(model) if model in AGENT_MODELS else -1
                current_model = AGENT_MODELS[(index + attempt)%len(AGENT_MODELS)]
                logger.debug(f"Inference attempt {attempt + 1}/{max_retries} using model {current_model}")
                raw_text=cls.make_request(messages,model=current_model, temperature=temperature)
                is_valid,error_msg=cls.is_valid_response(raw_text)
                if not(is_valid):
                    logger.warning(f"Invalid response from model {current_model}: {error_msg}")
                    raise Exception(error_msg)
                    
                next_thought, next_tool_name, next_tool_args,error_msg = cls.parse_response(raw_text)
                if error_msg:
                    logger.warning(f"Parse error for model {current_model}: {error_msg}")
                    raise Exception(error_msg)
                logger.info(f"Inference successful on attempt {attempt + 1} with model {current_model}")
                break
            except Exception as e:
                error_body = str(e)
                logger.warning(f"Inference attempt {attempt + 1}/{max_retries} failed: {error_body}")

                if attempt < max_retries - 1:
                    delay = base_delay

                    if "RATE_LIMIT_EXCEEDED" not in error_body and "RESERVED_TOKEN_PRESENT" not in error_body and "EMPTY_RESPONSE" not in error_body and  "TIMEOUT" not in error_body:
                        messages.append({"role":"assistant","content":raw_text})
                        messages.append({"role":"user","content":"observation: "+error_body})
                    sleep_time = random.uniform(1.2*delay, 1.5*delay)
                    logger.debug(f"Retrying after {sleep_time:.2f}s delay")
                    time.sleep(sleep_time)
                    continue
                else:
                    error_counter[cls.ErrorType.TIMEOUT.name]+=1
                    logger.error(f"All {max_retries} inference attempts failed. Last error: {error_body}")
                    raise RuntimeError(error_body)
        
        return next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages
    
    
    @classmethod
    def parse_malformed_json(cls,arguments:list[str], json_string:str)->dict | str:    
        # pattern of general json string with unescaped " in values keys from keys list
        pattern = ''
        for i, k in enumerate(arguments):
            pattern += f'"{k}": (.*)'
            if i != len(arguments) - 1:
                pattern += r',\s*'

        match=re.search(pattern, json_string)

        if not match:
            return f"Error: {json_string} can not match pattern {pattern}"
        
        result_json={}
        for i in range(len(arguments)):
            value=match.group(i+1)
            value=value.strip()
            if value.startswith('"') and value.endswith('"'):
                value=value[1:-1]
            #value=value.replace('"', '\\"')
            value=value.replace('\\n','\n')
            result_json[arguments[i]]=value
        return result_json
    
    @classmethod
    def parse_next_tool_args(cls,tool_name:str, next_tool_args: str)->dict | str:
        '''
        parse string to json, fix unecaped " in values like this: '{"a": "text "text2" text3 "text4"", "b": "text3"}'
        returns json or error message
        '''
        logger.debug(f"Parsing tool args for {tool_name}, args length: {len(next_tool_args)}")
        next_tool_args=next_tool_args.replace('```json','').strip('```')
        error_msg=''

        try:
            next_tool_args = Utils.load_json(next_tool_args.strip())
            logger.debug(f"Successfully parsed tool args for {tool_name}")
        except JSONDecodeError as e:
            logger.warning(f"JSON parsing failed for {tool_name}, attempting malformed JSON parser")
            error_msg=f"Invalid JSON: {next_tool_args}"    
            try:
                next_tool_args = cls.parse_malformed_json(EnhancedToolManager.get_tool_args_for_tool(tool_name,required_only=True), next_tool_args)
                logger.debug(f"Successfully parsed malformed JSON for {tool_name}")
            except EnhancedToolManager.Error as e:
                logger.error(f"Failed to parse tool args for {tool_name}: {e.message}")
                raise Exception(e.message)
            except Exception as e:
                logger.error(f"Failed to parse tool args for {tool_name}: {error_msg}")
                raise Exception(error_msg)
        return next_tool_args

    @classmethod
    def inference(cls, messages: List[Dict[str, Any]], model: str, run_id: str = str(uuid4()), temperature:float=0.0) -> dict:
        """Prod inference with caching"""
        logger.debug(f"Inference called with model={model}, temperature={temperature}, messages={len(messages)}")
        cleaned_msgs: List[Dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            if role not in {"system", "user", "assistant", "tool"}:
                continue
            content = m.get("content", "")

            if role == "assistant" and not content.strip():
                continue

            cleaned_msgs.append({"role": role, "content": content})

        if not cleaned_msgs:
            logger.error("No valid messages after cleaning")
            raise RuntimeError("No valid messages to send to proxy.")

        logger.debug(f"Cleaned messages: {len(cleaned_msgs)} messages")
        next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages = cls._request_next_action_with_retry(cleaned_msgs, model=model, temperature=temperature)
        logger.debug(f"Inference completed: tool_name={next_tool_name}, total_attempts={total_attempts}")
        return next_thought,next_tool_name,next_tool_args,raw_text,total_attempts,error_counter,messages
    
    @classmethod
    def sanitise_text_resp(cls,text_resp:str)->str:
        # remove all leading and trailing quotes
        text_resp=re.sub("[\'\"]*next_thought[\'\"]*:","next_thought:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_name[\'\"]*:","next_tool_name:",text_resp)
        text_resp=re.sub("[\'\"]*next_tool_args[\'\"]*:","next_tool_args:",text_resp)
        text_resp=re.sub("[\'\"]*observation[\'\"]*:","observation:",text_resp)
        if "next_thought" not in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:") and text_resp.find("next_tool_name:")>10:

            text_resp="next_thought: "+text_resp
        if "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            # remove all leading and trailing quotes in tool_name
            next_tool_name=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n").strip("\'").strip("\"").strip()
            text_resp=re.sub(f"next_tool_name:[\'\" ]*{next_tool_name}[\'\" ]*","next_tool_name: "+next_tool_name,text_resp)
        
        return text_resp

    @classmethod
    def parse_response(cls,text_resp: str)->tuple[str, Any, Any]:
        logger.debug(f"Parsing response, length: {len(text_resp)}")
        error_msg=None
        text_resp = text_resp.strip()
        text_resp=text_resp.split("observation:")[0]
        text_resp=text_resp.strip().strip("\n")
        text_resp=cls.sanitise_text_resp(text_resp)
        if "next_thought:" in text_resp and "next_tool_name:" in text_resp and "next_tool_args:" in text_resp and text_resp.find("next_thought:")<text_resp.find("next_tool_name:") and text_resp.find("next_tool_name:")<text_resp.find("next_tool_args:"):
            next_thought=text_resp.split("next_thought:")[1].split("next_tool_name:")[0].strip().strip("\n")
            next_tool_name_raw=text_resp.split("next_tool_name:")[1].split("next_tool_args:")[0].strip().strip("\n")
            next_tool_args_raw=text_resp.split("next_tool_args:")[1].strip().split("next_thought:")[0].strip().strip("\n")
            try:
                # Enforce arrays per new contract: if single string/object, wrap as arrays
                if next_tool_name_raw.startswith("["):
                    next_tool_name = Utils.load_json(next_tool_name_raw)
                else:
                    next_tool_name = [next_tool_name_raw]
                parsed_args = cls.parse_next_tool_args(next_tool_name, next_tool_args_raw)
                if isinstance(parsed_args, list):
                    next_tool_args = parsed_args
                else:
                    next_tool_args = [parsed_args for _ in next_tool_name]
                logger.debug(f"Successfully parsed response: thought length={len(next_thought)}, tool_name={next_tool_name}, tool_args count={len(next_tool_args)}")
            except JSONDecodeError as e:
                error_msg=f"Invalid JSON: {str(e)}"
                logger.warning(f"JSON decode error in parse_response: {error_msg}")
                
        else:
            if "next_thought:" not in text_resp:
                error_msg="Invalid response. next_thought not found"
            elif "next_tool_name:" not in text_resp:
                error_msg="Invalid response. next_tool_name not found"
            elif "next_tool_args:" not in text_resp:
                error_msg="Invalid response. next_tool_args not found"
            elif text_resp.find("next_thought:")>text_resp.find("next_tool_name:"):
                error_msg="Invalid response. next_thought is after next_tool_name"
            elif text_resp.find("next_tool_name:")>text_resp.find("next_tool_args:"):
                error_msg="Invalid response. next_tool_name is after next_tool_args"
            logger.warning(f"Parse response error: {error_msg}")

            return None,None,None,error_msg

        if len(next_tool_name) == 1:
            return next_thought, next_tool_name[0], next_tool_args[0], error_msg
            
        return next_thought, next_tool_name, next_tool_args,error_msg

class FunctionVisitor(ast.NodeVisitor):
    def __init__(self, file_content: str):
        self.functions = {}
        self.current_class = None
        self.class_hierarchy = []
        self.file_content = file_content

    def visit_ClassDef(self, node):
        self.class_hierarchy.append(node.name)
        self.current_class = "::".join(self.class_hierarchy)
        self.generic_visit(node)
        self.class_hierarchy.pop()
        self.current_class = "::".join(self.class_hierarchy) if self.class_hierarchy else None

    def _process_function(self, node):
        full_function_name = f"{self.current_class}::{node.name}" if self.current_class else node.name
        line_number = node.lineno
        if isinstance(node.decorator_list, list) and len(node.decorator_list) > 0:
            line_number = node.decorator_list[0].lineno
        
        end_line_number = line_number
        if isinstance(node.body, list) and len(node.body) > 0:
            end_line_number = node.body[-1].lineno
        
        lines = self.file_content.split("\n")
        body = "\n".join(lines[line_number-1:end_line_number])
        
        self.functions[full_function_name] = {
            "class": self.current_class,
            "body": body,
            "line_number": line_number
        }
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node):
        self._process_function(node)

    def visit_Module(self, node):
        self.current_class = None
        self.generic_visit(node)
        self.current_class = None

class Utils:
    
    @classmethod
    def count_tokens(cls, messages: list | str) -> int:
        '''Approximate GPT/LLM tokens using BPE-like estimation'''
        import re
        if isinstance(messages, list):
            text = " ".join(str(m.get("content", "") if isinstance(m, dict) else m) for m in messages)
        else:
            text = messages
        
        # Split into words and non-word tokens (punctuation, operators, etc.)
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text)
        
        count = 0
        for token in tokens:
            if token.isspace():
                continue  # Whitespace is typically absorbed
            elif len(token) == 1:
                count += 1  # Single chars (punctuation, operators)
            else:
                count += max(1, (len(token) + 2) // 3)
        
        return count


    @classmethod
    def limit_strings(cls,strings: str, n=1000)->str:
        '''
        Limit the number of strings to 1000
        '''
        strings_list=strings.split("\n")
        if len(strings_list)>n:
            return "\n".join(strings_list[:n])+"\n..." + f"({len(strings_list)-n} more lines)"
        else:
            return strings
    
    @classmethod
    def load_json(cls,json_string:str)->dict:
        try:
            return json.loads(json_string)
        except Exception as e:
            try:
                return eval(json_string)
            except Exception as e:
                logger.debug(f"JSON parsing failed, attempting LLM fix")
                fixed_json=EnhancedNetwork.fix_json_string_with_llm(json_string)
                if fixed_json:
                    logger.debug("JSON successfully fixed by LLM")
                    return fixed_json
                else:
                    logger.warning("JSON fix by LLM failed")
                    raise JSONDecodeError("Invalid JSON", json_string, 0)
                
class FixTaskEnhancedToolManager(EnhancedToolManager):

    def __init__(self, available_tools: Optional[list[str]] = [], test_runner: str = "pytest", test_runner_mode: str = "FILE", test_runner_test_file: str = "", repo_dir: str = ".", initial_checkpoint=None):
        self.new_files_created=[]
        self.is_solution_approved=False
        self.test_runner=test_runner
        self.test_runner_mode=test_runner_mode
        self.generated_test_files=[]
        self.initial_checkpoint=initial_checkpoint
        self.latest_generated_patch=""
        self.observation_dir = ".observation"
        self.saved_observation_counter = 0
        self.test_runner_test_file = test_runner_test_file
        self.repo_dir = repo_dir
        self.TOOL_LIST = {}
        # Create observation directory if it doesn't exist
        os.makedirs(self.observation_dir, exist_ok=True)
        # Check all classes in the method resolution order (MRO) to include inherited tools
        for cls in self.__class__.__mro__:
            for name, attr in cls.__dict__.items():
                if getattr(attr, "is_tool", False) and name not in self.TOOL_LIST:
                    if available_tools is not None and name not in available_tools: # if available_tools is provided, only include tools in the list
                        continue
                    self.TOOL_LIST[name] = self.__class__.tool_parsing(attr)
                
        self.tool_failure={ k:{j:0 for j in self.Error.ErrorType.__members__} for k in self.TOOL_LIST.keys() }

        self.tool_invocations={ k:0 for k in self.TOOL_LIST.keys() }

    def check_syntax_error(self,content:str,file_path:str="<unknown>")->bool:
        try:
            ast.parse(content, filename=file_path)
            return False, None
        except SyntaxError as e:
            logger.debug(f"Syntax error detected in {file_path}: {e}")
            return True, f"Syntax error. {str(e)}"

    def _get_file_content(self, file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None, limit: int = 1000, add_line_numbers: bool = False, structural_truncation: bool = False) -> str:
        def add_line_numbers_to_content(content: str, start_line: int = 1) -> str:
            """Helper method to add line numbers to content."""
            lines = content.splitlines()
            numbered_lines = []
            for i, line in enumerate(lines):
                line_num = start_line + i
                numbered_lines.append(f"{line_num:6}|{line}")
            return '\n'.join(numbered_lines)
        
        def format_class_skeleton(node, source: str) -> str:
            """Format class with method signatures."""
            import ast
            lines = source.splitlines()
            result = [f"\n## Line {node.lineno}: class {node.name}:"]
            
            # Get docstring if exists
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split('\n')[0]
                result.append(f'    """{first_line}..."""\n')
            
            # List all methods
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [arg.arg for arg in item.args.args]
                    result.append(f"    Line {item.lineno}: def {item.name}({', '.join(args)})")
            
            return "\n".join(result) + "\n"
        
        def format_function_skeleton(node, source: str) -> str:
            """Format function signature with docstring."""
            import ast
            args = [arg.arg for arg in node.args.args]
            signature = f"Line {node.lineno}: def {node.name}({', '.join(args)})"
            
            docstring = ast.get_docstring(node)
            if docstring:
                first_line = docstring.split('\n')[0]
                return f"{signature}\n    '''{first_line}...'''\n"
            return f"{signature}\n"
        
        def show_head_tail_summary(lines: list, file_path: str) -> str:
            """Show first 500 and last 500 lines with summary in middle."""
            total = len(lines)
            head = lines[:500]
            tail = lines[-500:]
            
            result = f"=== {file_path} ({total} lines) ===\n\n"
            result += "=== First 500 lines ===\n"
            result += "".join(head)
            result += f"\n\n... [{total - 1000} lines omitted] ...\n\n"
            result += "=== Last 500 lines ===\n"
            result += "".join(tail)
            result += "\n\nTIP: Use get_file_content with start_line/end_line to view specific sections"
            
            return result
        
        def get_file_skeleton(file_path: str) -> str:
            """
            Generate a skeleton view showing:
            - All imports
            - All class definitions with their methods (signatures only)
            - All top-level functions (signatures + docstrings)
            - Important constants/globals
            """
            if not file_path.endswith('.py'):
                # For non-Python files, show first/last N lines with middle summary
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                return show_head_tail_summary(lines, file_path)
            
            import ast
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    source = f.read()
                tree = ast.parse(source)
                
                skeleton_parts = []
                skeleton_parts.append(f"=== Skeleton view of {file_path} ({len(source.splitlines())} lines) ===\n")
                skeleton_parts.append("TIP: Use get_file_content with start_line/end_line to view specific sections\n\n")
                
                # Extract imports
                imports = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"Line {node.lineno}: import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ''
                        names = ', '.join(a.name for a in node.names)
                        imports.append(f"Line {node.lineno}: from {module} import {names}")
                
                if imports:
                    skeleton_parts.append("## Imports:\n" + "\n".join(imports[:20]))
                    if len(imports) > 20:
                        skeleton_parts.append(f"... and {len(imports) - 20} more imports")
                    skeleton_parts.append("\n\n")
                
                # Extract classes and functions
                for node in tree.body:
                    if isinstance(node, ast.ClassDef):
                        skeleton_parts.append(format_class_skeleton(node, source))
                    elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                        skeleton_parts.append(format_function_skeleton(node, source))
                    elif isinstance(node, ast.Assign):
                        # Show important constants
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id.isupper():
                                skeleton_parts.append(f"Line {node.lineno}: {target.id} = ...\n")
                
                return "\n".join(skeleton_parts)
                
            except Exception as e:
                logger.warning(f"Failed to parse {file_path} for skeleton: {e}")
                # Fallback to head/tail
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                return show_head_tail_summary(lines, file_path)
        
        # If search term is provided, use specialized search
        if search_term:
            logger.debug(f"search_term specified: {search_term}, searching in v2")
            return self.search_in_specified_file_v2(file_path, search_term)
        
        # Check if file is too large and user didn't specify range
        if search_start_line is None and search_end_line is None and structural_truncation == True:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                line_count = sum(1 for _ in f)
            
            if line_count > 1000:
                # Show skeleton view instead of truncated content
                return get_file_skeleton(file_path)

        # Adjust start/end lines if they fall within a function
        if file_path.endswith(".py"):
            func_ranges = self.get_function_ranges(file_path)

            if search_start_line is not None:
                for start, end, name in func_ranges:
                    if start <= search_start_line <= end and start < search_start_line:
                        logger.debug(f"Adjusting start line {search_start_line} to {start} (function {name})")
                        search_start_line = start

            if search_end_line is not None:
                for start, end, name in func_ranges:
                    if start <= search_end_line <= end and end > search_end_line:
                        logger.debug(f"Adjusting end line {search_end_line} to {end} (function {name})")
                        search_end_line = end

        logger.debug(f"search start line: {search_start_line}, search end line: {search_end_line}")

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            if search_start_line is not None or search_end_line is not None:
                lines = f.readlines()
                start_idx = max(0, (search_start_line or 1) - 1)
                end_idx = min(len(lines), search_end_line or len(lines))
                content = "".join(lines[start_idx:end_idx])
                if add_line_numbers:
                    numbered_content = add_line_numbers_to_content(content, start_idx + 1)
                    result = f"Lines {start_idx+1}-{end_idx} of {file_path}:\n{numbered_content}"
                else:
                    result = f"Lines {start_idx+1}-{end_idx} of {file_path}:\n{content}"
            else:
                content = f.read()
                if add_line_numbers:
                    numbered_content = add_line_numbers_to_content(content, 1)
                    result = numbered_content
                else:
                    result = content

        return Utils.limit_strings(result, n=limit) if limit != -1 else result

    def _save_large_observation(self, observation: str, tool_name: str) -> str:
        """
        Save a large observation to a file in .observation directory and return the file path.
        """
        self.saved_observation_counter += 1
        filename = f"observation_{self.saved_observation_counter}_{tool_name}_{int(time.time())}.txt"
        file_path = os.path.join(self.observation_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(observation)
            logger.info(f"Saved large observation to {file_path} ({len(observation)} characters)")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save observation to {file_path}: {e}")
            return f"Error: Failed to save observation: {e}"
 
    def _save(self,file_path: str, content: str)->str:
        logger.debug(f"Saving file: {file_path}, content length: {len(content)}")
        is_syntax_error, error = self.check_syntax_error(content)
        if not is_syntax_error:
            with open(file_path, "w") as file:
                file.write(content)
            self.new_files_created.append(file_path)
            return f"File {file_path} saved successfully"
        else:
            return f"Error saving file: {error}"

    def get_function_ranges(self,file_path: str)->list[tuple[int, int, str]]:
        # Try to parse the file to map lines to their enclosing functions.
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return 0, 0, f"Error reading '{file_path}': {e}"
        try:
            tree = ast.parse("\n".join(source_lines), filename=file_path)
        except SyntaxError as e:
            return 0, 0, f"Error parsing '{file_path}': {e}, {traceback.format_exc()}"

        func_ranges: list[tuple[int, int, str]] = []  # (start, end, name)
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    start = getattr(node, 'lineno', None)
                    end = getattr(node, 'end_lineno', None)
                    if start is not None and end is not None:
                        func_ranges.append((start, end, node.name))
        return func_ranges

    def _extract_function_matches(self,file_path: str, search_term: str, *, max_output_lines: int = 1000) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_lines = f.read().splitlines()
        except Exception as e:
            return f"Error reading '{file_path}': {e}"

        # Identify all lines that contain the search term.
        match_lines = [idx + 1 for idx, line in enumerate(source_lines) if search_term in line]
        if not match_lines:
            return f"'{search_term}' not found in file '{file_path}'"

        func_ranges=self.get_function_ranges(file_path)

        def _containing_function(line_no: int):
            for start, end, name in func_ranges:
                if start <= line_no <= end:
                    return (start, end, name)
            return None

        functions_to_return: list[tuple[int, int, str]] = []
        standalone_lines: list[int] = []
        for ln in match_lines:
            info = _containing_function(ln)
            if info and info not in functions_to_return:
                functions_to_return.append(info)
            elif not info:
                standalone_lines.append(ln)

        chunks: list[str] = []
        for start, end, name in functions_to_return:
            func_src = "\n".join(source_lines[start - 1:end])
            chunks.append(f"(lines {start}-{end}):\n{func_src}")

        for ln in standalone_lines:
            chunks.append(f"{ln}:{source_lines[ln - 1]}")

        return Utils.limit_strings("\n\n".join(chunks), n=max_output_lines)

    def _truncate_output(self, output: str, max_first_lines: int = 500, max_last_lines: int = 500) -> str:
        """Truncate long output to first N and last N lines with summary in middle."""
        lines = output.split('\n')
        total_lines = len(lines)
        
        if total_lines <= max_first_lines + max_last_lines:
            return output
        
        first_lines = lines[:max_first_lines]
        last_lines = lines[-max_last_lines:]
        omitted_lines = total_lines - max_first_lines - max_last_lines
        
        truncated = '\n'.join(first_lines)
        truncated += f"\n\n... ({omitted_lines} lines omitted) ...\n\n"
        truncated += '\n'.join(last_lines)
        
        return truncated
    
    def _run_repo_tests(self, file_paths: List[str]) -> str:
        def truncate_output(output: str, max_first_lines: int = 500, max_last_lines: int = 500) -> str:
            """Truncate long output to first N and last N lines with summary in middle."""
            lines = output.split('\n')
            total_lines = len(lines)
            
            if total_lines <= max_first_lines + max_last_lines:
                return output
            
            first_lines = lines[:max_first_lines]
            last_lines = lines[-max_last_lines:]
            omitted_lines = total_lines - max_first_lines - max_last_lines
            
            truncated = '\n'.join(first_lines)
            truncated += f"\n\n... ({omitted_lines} lines omitted) ...\n\n"
            truncated += '\n'.join(last_lines)
            
            return truncated
        if len(file_paths) == 0:
            return "Error: No files provided to run the tests for."
        
        if self.test_runner == "pytest":
            file_paths_str = ", ".join([f"'{f}'" for f in file_paths])
            cmd = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str, version_compatibility_fix=VERSION_COMPATIBILITY_FIX)
            print(f"Running command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
            output = (result.stdout or "") + (result.stderr or "")
        elif self.test_runner == "unittest":
            output = ""
            for file_path in file_paths:
                result = subprocess.run(
                    ["python", file_path],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                current_output = (result.stdout or "") + (result.stderr or "")
                output += current_output
        else:
            if self.test_runner_mode == "MODULE":
                modules = [filepath_to_module(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(modules)}"
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
            else:
                files_to_test = [clean_filepath(f, os.getcwd(), self.test_runner) for f in file_paths]
                cmd = f"{self.test_runner} {' '.join(files_to_test)}"
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
            if self._check_dependency_errors(output):
                file_paths_str = ", ".join([f"'{f}'" for f in file_paths])
                cmd = PYTEST_COMMAND_TEMPLATE.format(file_paths=file_paths_str, version_compatibility_fix=VERSION_COMPATIBILITY_FIX)
                print(f"Running command: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=150)
                output = (result.stdout or "") + (result.stderr or "")
        return truncate_output(output)
    
    def get_final_git_patch(self) -> str:
        logger.debug("Generating final git patch")
        try:
            exts = (".py", ".ini", ".cfg", ".toml")
            exclude = {"src/agent.py", "src/agent_runner.py"}
            try:
                for _p in getattr(self, "generated_test_files", []):
                    exclude.add(os.path.relpath(_p))
            except Exception:
                pass
            
            # Exclude all files in .observation directory
            observation_dir = getattr(self, "observation_dir", ".observation")
            if os.path.exists(observation_dir):
                try:
                    for root, dirs, files in os.walk(observation_dir):
                        for file in files:
                            file_path = os.path.relpath(os.path.join(root, file))
                            exclude.add(file_path)
                except Exception:
                    pass
            logger.debug(f"Excluding files from patch: {exclude}")
            ls = subprocess.run(
                ["git", "ls-files", "-m", "-o", "--exclude-standard"],
                capture_output=True, text=True, timeout=30, check=True
            ).stdout.splitlines()
            to_add = [f for f in ls if f.endswith(exts) and f not in exclude]
            logger.info(f"Found {len(to_add)} files to add to git patch")
            if to_add:
                subprocess.run(["git", "add", "--"] + to_add, check=True, timeout=30)
            diff = subprocess.run(
                ["git", "diff", "--cached", "--no-color", "--unified=3"],
                capture_output=True, text=True, timeout=30, check=True
            )
            patch_text = diff.stdout or ""
            logger.info(f"Generated git patch with length: {len(patch_text)}")
            return patch_text
        except Exception as e:
            logger.error(f"Error generating git patch: {e}")
            return f"Error generating git patch: {e}"

    @EnhancedToolManager.tool
    def generate_tests(self, file_path: str, test_code: str, position: str = "append") -> str:
        '''
        Create or append tests to the specified test file. Supports inserting entire test classes, standalone test functions, and other supporting code blocks (imports, helpers). Generated tests are excluded from the final patch automatically. For unittest files, automatically detects TestCase classes and can merge new test methods into existing classes with proper indentation.
        Arguments:
            file_path: path to the test file to create or modify
            test_code: the full test code block to insert (class(es), function(s), and/or helpers)
            position: where to place the code: "append", "top", "after_imports", "before_main", or "auto"
                     - "auto" (recommended): tries class merge, then unittest method insertion, then before_main,
                       then after_imports, finally append
                     - "append": tries unittest class insertion first, then appends to end of file
        '''
        if not file_path.endswith('.py'):
            return f"Error: file '{file_path}' is not a python file."
        if "_pass_original.py" in file_path:
            return "ERROR: you cannot apply code edit to a test file that is already passing."

        # Ensure directory exists
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        # Normalize newline handling
        test_fn = (test_code or "").strip()
        if not test_fn:
            return "Error: test_function_code cannot be empty."

        is_new_file = not os.path.exists(file_path)

        def _parse_classes_and_functions(block: str):
            try:
                tree = ast.parse(block)
                lines = block.splitlines()
                classes = []
                functions = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        start = node.lineno - 1
                        end = (node.end_lineno or node.lineno) - 1
                        classes.append((node, "\n".join(lines[start:end+1])))
                    elif isinstance(node, ast.FunctionDef):
                        start = node.lineno - 1
                        end = (node.end_lineno or node.lineno) - 1
                        functions.append((node, "\n".join(lines[start:end+1])))
                return classes, functions
            except Exception:
                return [], []

        def _insert_into_unittest_class(content: str, block: str) -> str:
            try:
                tree = ast.parse(content)
                lines = content.splitlines()
                
                # Find unittest.TestCase classes
                test_classes = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Check if it inherits from unittest.TestCase or TestCase
                        for base in node.bases:
                            base_name = ""
                            if isinstance(base, ast.Name):
                                base_name = base.id
                            elif isinstance(base, ast.Attribute):
                                base_name = base.attr
                            
                            if "TestCase" in base_name:
                                test_classes.append(node)
                                break
                
                if not test_classes:
                    return None  # No unittest class found
                
                # Use the first test class found
                test_class = test_classes[0]
                
                # Find the last line of the class (before any standalone functions or if __name__)
                class_end_line = test_class.end_lineno - 1  # 0-indexed
                
                # Find the last method in the class to get proper indentation
                last_method_line = test_class.lineno  # Start of class
                for node in test_class.body:
                    if isinstance(node, ast.FunctionDef):
                        last_method_line = max(last_method_line, node.end_lineno - 1)
                
                # Get indentation from the class (typically 4 spaces)
                class_indent = ""
                for i in range(test_class.lineno - 1, len(lines)):
                    line = lines[i]
                    if line.strip() and line.strip().startswith("def "):
                        # Found a method, extract its indentation
                        class_indent = line[:len(line) - len(line.lstrip())]
                        break
                
                if not class_indent:
                    class_indent = "    "  # Default to 4 spaces
                
                # Prepare the test method with proper indentation
                # First dedent the block to remove any existing indentation
                dedented_block = textwrap.dedent(block)
                test_method_lines = dedented_block.split('\n')
                
                # Now add proper class-level indentation
                indented_method = []
                for line in test_method_lines:
                    if line.strip():  # Non-empty line
                        # Add class indentation to every line
                        indented_method.append(class_indent + line)
                    else:
                        indented_method.append("")  # Keep empty lines
                
                # Insert the method at the end of the class
                new_lines = lines[:class_end_line + 1]
                new_lines.append("")  # Blank line before new method
                new_lines.extend(indented_method)
                new_lines.extend(lines[class_end_line + 1:])
                
                return "\n".join(new_lines)
                
            except Exception as e:
                print(f"Error inserting into unittest class: {e}")
                return None

        def _merge_classes_into_existing(content: str, block: str) -> str:
            try:
                existing_tree = ast.parse(content)
                content_lines = content.splitlines()
                new_classes, _ = _parse_classes_and_functions(block)
                if not new_classes:
                    return None

                # Map existing classes by name
                existing_classes = {}
                for node in ast.walk(existing_tree):
                    if isinstance(node, ast.ClassDef):
                        existing_classes[node.name] = node

                updated = content
                for cls_node, cls_src in new_classes:
                    # Check if new class is a unittest.TestCase subclass
                    is_test_case = False
                    for base in cls_node.bases:
                        base_name = getattr(base, 'id', None) or getattr(base, 'attr', None) or ""
                        if "TestCase" in str(base_name):
                            is_test_case = True
                            break
                    if not is_test_case:
                        continue

                    if cls_node.name not in existing_classes:
                        # No merge target for this class
                        continue

                    # Extract methods from new class
                    block_lines = cls_src.splitlines()
                    new_methods_sources = []
                    for member in cls_node.body:
                        if isinstance(member, ast.FunctionDef):
                            m_start = member.lineno - 1
                            m_end = (member.end_lineno or member.lineno) - 1
                            method_src = "\n".join(block_lines[m_start:m_end+1])
                            new_methods_sources.append(method_src)

                    if not new_methods_sources:
                        continue

                    # Compute insertion point and indentation in existing class
                    target_cls = existing_classes[cls_node.name]
                    class_end_line = target_cls.end_lineno - 1
                    last_method_line = target_cls.lineno
                    for n in target_cls.body:
                        if isinstance(n, ast.FunctionDef):
                            last_method_line = max(last_method_line, n.end_lineno - 1)
                    # Determine indentation from first method line inside class
                    class_indent = "    "
                    for i in range(target_cls.lineno - 1, len(content_lines)):
                        line = content_lines[i]
                        if line.strip().startswith("def "):
                            class_indent = line[:len(line) - len(line.lstrip())]
                            break

                    # Build indented methods
                    indented_methods = []
                    for m_src in new_methods_sources:
                        dedented = textwrap.dedent(m_src)
                        for ln in dedented.splitlines():
                            indented_methods.append((class_indent + ln) if ln.strip() else "")
                        indented_methods.append("")

                    # Insert before end of class
                    lines = updated.splitlines()
                    new_lines = lines[:class_end_line + 1] + [""] + indented_methods + lines[class_end_line + 1:]
                    updated = "\n".join(new_lines)

                if updated != content:
                    return updated
                return None
            except Exception as e:
                print(f"Error merging classes: {e}")
                return None

        def _insert_after_imports(content: str, block: str) -> str:
            lines = content.splitlines()
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    insert_idx = i + 1
                elif stripped == "" or stripped.startswith("#"):
                    # allow header comments/blank lines before imports
                    insert_idx = max(insert_idx, i + 1)
                else:
                    break
            lines = lines[:insert_idx] + (["", block, ""] if insert_idx < len(lines) else ["", block]) + lines[insert_idx:]
            return "\n".join(lines).rstrip() + "\n"

        def _insert_before_main(content: str, block: str) -> str:
            marker = "if __name__ == \"__main__\":"
            idx = content.find(marker)
            if idx == -1:
                return None
            return content[:idx].rstrip() + "\n\n" + block + "\n\n" + content[idx:]

        if is_new_file:
            new_content = test_fn + "\n"
            # Validate standalone content before writing
            is_err, err = self.check_syntax_error(new_content)
            if is_err:
                return f"Error: generated test function has syntax error: {err}"
        else:
            original = self._get_file_content(file_path, limit=-1)
            # Avoid duplicating exact same function text
            if test_fn in original:
                rel = os.path.relpath(file_path)
                if rel not in self.generated_test_files:
                    self.generated_test_files.append(rel)
                return f"Test already present in '{rel}', no changes made."

            # Build candidate insertion strategies in order
            candidates = []
            if position == "append":
                candidates = [
                    lambda src: _merge_classes_into_existing(src, test_fn) or _insert_into_unittest_class(src, test_fn),  # Try class merge, then unittest method first
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n"
                ]
            elif position == "top":
                candidates = [lambda src: test_fn + "\n\n" + src]
            elif position == "after_imports":
                candidates = [lambda src: _insert_after_imports(src, test_fn)]
            elif position == "before_main":
                candidates = [lambda src: (_insert_before_main(src, test_fn) or src.rstrip() + "\n\n" + test_fn + "\n")]
            elif position == "auto":
                candidates = [
                    lambda src: _merge_classes_into_existing(src, test_fn),  # FIRST: Try to merge into existing TestCase classes
                    lambda src: _insert_into_unittest_class(src, test_fn),   # THEN: Try to insert as test method
                    lambda src: (_insert_before_main(src, test_fn) or _insert_after_imports(src, test_fn)),
                    lambda src: src.rstrip() + "\n\n" + test_fn + "\n",
                    lambda src: test_fn + "\n\n" + src,
                ]
            else:
                return f"Error: invalid position '{position}'. Use 'append', 'top', 'after_imports', 'before_main', or 'auto'."

            # Try each candidate until one passes syntax check
            new_content = None
            first_error = None
            for builder in candidates:
                try:
                    candidate = builder(original)
                    is_err, err = self.check_syntax_error(candidate)
                    if not is_err:
                        new_content = candidate
                        break
                    if first_error is None:
                        first_error = err
                except Exception as e:
                    if first_error is None:
                        first_error = e
                    continue

            if new_content is None:
                return f"Error: inserting test caused syntax error. First error: {first_error}"

        self._save(file_path, new_content)

        # Track for exclusion from final patch
        rel = os.path.relpath(file_path)
        if rel not in self.generated_test_files:
            self.generated_test_files.append(rel)

        # Find where the test code was inserted and show context
        return f"Tests created in '{rel}' (position={position})."

    @EnhancedToolManager.tool
    def get_file_content(self,file_path: str, search_start_line: int = None, search_end_line: int = None, search_term: str = None)->str:
        '''
        Retrieves file contents with optional filtering based on search term and line numbers
        Arguments:
            file_path: filesystem path to target file. This file must be python file.
            search_start_line: optional start line number to begin extraction (1-indexed)
            search_end_line: optional end line number to end extraction (1-indexed)
            search_term: optional text pattern to filter matching lines
        '''
        logger.debug(f"Getting file content: {file_path}, lines={search_start_line}-{search_end_line}, search_term={search_term}")
        result = self._get_file_content(file_path,search_start_line,search_end_line,search_term,limit=5000, structural_truncation=True)
        logger.debug(f"Retrieved file content, length: {len(result)}")
        return result
    
    @EnhancedToolManager.tool
    def list_directory(self, root: str = ".", max_depth: int = 1) -> str:
        """
        Lists the directory structure (files + folders) from `root` up to `max_depth`.

        Arguments:
            root: Directory to list (default ".")
            max_depth: Maximum depth of recursion (default 1)
        """
        logger.debug(f"Listing directory structure: root={root}, max_depth={max_depth}")
        ignore = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules',
            '.tox', '.venv', 'venv', '.mypy_cache', '.eggs', 'build', 'dist'
        }

        def tree(path: str, prefix: str = "", depth: int = 0) -> list[str]:
            if depth > max_depth:
                return []

            try:
                items = sorted(os.listdir(path))
            except Exception:
                return [f"{prefix}[Error reading directory]"]

            # separate dirs and files
            dirs = [
                i for i in items
                if os.path.isdir(os.path.join(path, i))
                and not i.startswith(".")
                and i not in ignore
                and not i.endswith(".egg-info")
            ]

            files = [
                i for i in items
                if os.path.isfile(os.path.join(path, i))
                and not i.startswith(".")
            ]

            lines: list[str] = []

            # add directories
            for idx, d in enumerate(dirs):
                is_last = (idx == len(dirs) - 1) and not files
                branch = "└── " if is_last else "├── "
                new_prefix = prefix + ("    " if is_last else "│   ")

                lines.append(f"{prefix}{branch}{d}/")
                lines.extend(tree(os.path.join(path, d), new_prefix, depth + 1))

            # add files
            for idx, f in enumerate(files):
                is_last = idx == len(files) - 1
                branch = "└── " if is_last else "├── "
                lines.append(f"{prefix}{branch}{f}")

            return lines

        # build final tree
        entries = tree(root, "", 0)

        result = f"Directory structure (depth={max_depth}):\n{root}/\n" + "\n".join(entries)
        logger.debug(f"Directory structure listed: {len(entries)} entries")
        return result
 
    @EnhancedToolManager.tool
    def save_file(self,file_path: str, content: str)->str:
        '''
        Writes text content to specified filesystem location. If there are any syntax errors in the code, it rejects the edit with an error message. Do not use this tool to create test or files to reproduce the error.
        Arguments:
            file_path: target filesystem path
            content: text data to write
        '''
        logger.info(f"Saving file: {file_path}, content length: {len(content)}")
        if "test" in file_path.lower() or "reproduce" in file_path.lower():
            logger.warning(f"Attempted to save test/reproduce file: {file_path}")
            return f"Error: You cannot use this tool to create test or files to reproduce the error."
        result = self._save(file_path, content)
        logger.info(f"File saved successfully: {file_path}")
        return result
    
    @EnhancedToolManager.tool
    def search_in_all_files_content(self, search_term: str, case_sensitive: bool = False) -> str:
        '''
        Search for a text pattern across all .py files in the project, excluding any file with "test" in its path.
        Use at the beginning of the workflow to locate all possible references to a function, class, or variable.

        Arguments:
            search_term: text pattern to locate (e.g., "def test_function", "*SomeClass*")
            case_sensitive: flag to determine if the search should be case-sensitive
        '''
        logger.info(f"Searching all files for: '{search_term}' (case_sensitive={case_sensitive})")
        output = []
        search_flags = 0 if case_sensitive else re.IGNORECASE

        # Walk through all directories and find Python files
        for root, _, files in os.walk("."):
            # Skip .git and docs directories
            if ".git" in root or "docs" in root:
                continue

            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)

                    # Always check if search term is in the file name
                    if re.search(search_term, file_path, search_flags):
                        output.append(f"{file_path} | Filename match")

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        if not re.search(search_term, content, search_flags):
                            continue

                        # Parse the file content using AST
                        tree = ast.parse(content, filename=file_path)
                        visitor = FunctionVisitor(content)
                        visitor.visit(tree)

                        for function_name, function_info in visitor.functions.items():
                            body = function_info["body"]
                            if re.search(search_term, body, search_flags):
                                lines = body.split("\n")
                                for idx, line in enumerate(lines):
                                    if re.search(search_term, line, search_flags):
                                        line_number = function_info["line_number"] + idx
                                        output.append(f"{file_path}:{line_number} | {function_name} | {line.rstrip()}")
                    except Exception as e:
                        pass

        output = Utils.limit_strings("\n".join(output), n=100)
        if not output:
            logger.info(f"Search term '{search_term}' not found in codebase")
            return f"'{search_term}' not found in the codebase."
        logger.info(f"Found {len(output.splitlines())} matches for '{search_term}'")

        if Utils.count_tokens(output) > MAX_TOOL_CALL_RESPONSE_TOKENS:
            return f"Error: search result exceeded token limit ({Utils.count_tokens(output)} tokens > {MAX_TOOL_CALL_RESPONSE_TOKENS} tokens limit). Make your search term more specific please"

        return output

    @EnhancedToolManager.tool
    def search_in_specified_file_v2(self,file_path: str, search_term: str)->str:
        '''
        Locates text patterns within a specific file
        Arguments:
            file_path: target file for pattern matching. This file must be python file.
            search_term: text pattern to find (e.g., "def test_function", "*SomeClass*")
        '''
        logger.debug(f"Searching in file {file_path} for: '{search_term}'")
        if not file_path.endswith(".py"):
            logger.warning(f"Attempted to search in non-Python file: {file_path}")
            return f"Error: file '{file_path}' is not a python file."
        result = self._extract_function_matches(file_path, search_term)
        logger.debug(f"Search completed, result length: {len(result)}")

        if Utils.count_tokens(result) > MAX_TOOL_CALL_RESPONSE_TOKENS:
            return f"Error: search result exceeded token limit ({Utils.count_tokens(result)} tokens > {MAX_TOOL_CALL_RESPONSE_TOKENS} tokens limit). Make your search term more specific or target a smaller file range."

        return result
    
    @EnhancedToolManager.tool
    def run_repo_tests(self,file_paths:List[str])->str:
        '''
        Runs the tests for the repository. This tool will only run the tests for the files provided.
        Arguments:
            file_paths: path of the files to run the tests for.
        Output:
            Returns the stdout/stderr from the executed files.
        '''
        def summarize_test_results(output: str) -> str:
            """Summarize the test results."""
            summary_test_results_prompt = textwrap.dedent("""
            You are a python test result analyzer. You will be provided with the raw output of the tests. Your task is to analyze the test results and provide a summary of the test results including:
            - Total number of tests
            - Total number of passed tests
            - Total number of failed tests
                - For each failed test, provide the test name, error message, the root cause of the failure, and traceback of the failure.
                - Gropu the failures by the same root cause of the failure.
            - Total number of errored tests
                - For each errored test, provide the test name, error message, and the root cause of the error.
                - Gropu the errored tests by the same root cause of the error.
            **DO NOT INCLUDE xfailed tests, and skipped tests in the summary.**

            Here is the test results:
            {output}
            """).strip()
            try:
                raw_text = EnhancedNetwork.make_request(messages=[{"role": "user", "content": summary_test_results_prompt.format(output=output)}], model=KIMI_MODEL_NAME)
                return raw_text
            except Exception as e:
                logger.error(f"Exception in summarize_test_results: {e}")
                return "Error: Test result are too big. Please try again with a smaller test suite."

        output = self._run_repo_tests(file_paths)
        # return output
        if Utils.count_tokens(output) > MAX_TOOL_CALL_RESPONSE_TOKENS:
            logger.info("Test output is too big, summarizing the test results")
            return summarize_test_results(output)
        else:
            return output

    @EnhancedToolManager.tool
    def apply_code_edit(self, file_path: str, search: str, replace: str) -> str:
        '''
        Performs targeted text replacement within source files. If there are any syntax errors in the code, it rejects the edit with an error message. Please note use you can only use this tool after you have approval from user on your proposed solution.
        Arguments:
            file_path: target file for modification
            search: exact text pattern to locate and replace
            replace: new text content to substitute    
        Output:
            operation status - success confirmation or detailed error with guidance
        '''
        logger.info(f"Applying code edit to {file_path}, search length: {len(search)}, replace length: {len(replace)}")
        def add_context_to_similar_match(original_content: str, formatted_match: str, context_lines: int = 2) -> str:
            """Add context lines around a similar match for better understanding."""
            lines = original_content.split('\n')
            
            # Extract the actual content from the formatted match (remove the description part)
            match_lines = formatted_match.split('\n')
            if len(match_lines) < 2:
                return formatted_match
                
            # Skip the description line (e.g., "Lines 45-47: ..." or "Line 23: ...")
            actual_content_lines = match_lines[1:]
            actual_content = '\n'.join(actual_content_lines)
            
            # Find where this content appears in the original file
            best_match_start = -1
            best_similarity = 0
            
            # Search for the best matching position in the original content
            for i in range(len(lines) - len(actual_content_lines) + 1):
                candidate_lines = lines[i:i + len(actual_content_lines)]
                candidate_content = '\n'.join(candidate_lines)
                
                import difflib
                similarity = difflib.SequenceMatcher(None, actual_content.strip(), candidate_content.strip()).ratio()
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_start = i
            
            if best_match_start == -1:
                return formatted_match  # Fallback to original if can't find position
            
            # Calculate context boundaries
            start_line = max(0, best_match_start - context_lines)
            end_line = min(len(lines), best_match_start + len(actual_content_lines) + context_lines)
            
            # Build the context with line numbers
            context_lines_list = []
            for i in range(start_line, end_line):
                line_num = i + 1
                prefix = ">>> " if best_match_start <= i < best_match_start + len(actual_content_lines) else "    "
                context_lines_list.append(f"{prefix}{line_num:4}| {lines[i]}")
            
            # Extract original description
            description = match_lines[0] if match_lines else f"Match found at lines {best_match_start+1}-{best_match_start+len(actual_content_lines)}"
            
            return f"{description}\n" + "\n".join(context_lines_list)

        def find_most_similar_content(original_content: str, search_string: str, max_results: int = 3) -> list[tuple[float, str]]:
            """Find the most similar content chunks to the search string."""
            import difflib
            
            # Split content into meaningful chunks
            lines = original_content.split('\n')
            
            # Try different chunk sizes to find the best match
            chunks = []
            
            # Individual lines
            for i, line in enumerate(lines):
                if line.strip():  # Skip empty lines
                    chunks.append((f"Line {i+1}: {line.strip()}", line.strip()))
            
            # Multi-line chunks (3-5 lines) for better context
            search_lines = search_string.split('\n')
            target_chunk_size = max(3, len(search_lines))
            
            for i in range(len(lines) - target_chunk_size + 1):
                chunk_lines = lines[i:i + target_chunk_size]
                chunk_content = '\n'.join(chunk_lines).strip()
                if chunk_content:
                    chunks.append((f"Lines {i+1}-{i+target_chunk_size}: ...", chunk_content))
            
            # Calculate similarity scores
            similarities = []
            for chunk_desc, chunk_content in chunks:
                ratio = difflib.SequenceMatcher(None, search_string.strip(), chunk_content).ratio()
                if ratio > 0.3:  # Only include reasonably similar content
                    similarities.append((ratio, chunk_desc, chunk_content))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [(ratio, f"{desc}\n{content}") for ratio, desc, content in similarities[:max_results]]
        
        if search == replace:
            logger.warning("Attempted to apply code edit with identical search and replace")
            return "ERROR: search and replace are the same. Please provide a different search and replace."
        if not os.path.exists(file_path):
            logger.warning(f"Attempted to edit non-existent file: {file_path}")
            return f"Error: file '{file_path}' does not exist."
        
        original=self._get_file_content(file_path,limit=-1)

        match original.count(search):
            case 0:
                # Find most similar content to help LLM correct the search string
                similar_matches = find_most_similar_content(original, search, 1)
                
                error_msg = f"Error: search string not found in file {file_path}."
                
                if similar_matches:
                    error_msg += f"\n\nMost similar snippet found (you may need to adjust your search string):"
                    for i, (ratio, content) in enumerate(similar_matches, 1):
                        similarity_pct = int(ratio * 100)
                        # Add context lines around the match for better understanding
                        content_with_context = add_context_to_similar_match(original, content, context_lines=2)
                        error_msg += f"\n\n{i}. Similarity: {similarity_pct}%\n{content_with_context}"
                else:
                    error_msg += " No similar content found. Please check the file content and provide the exact code you want to replace."
                
                return error_msg
            case 1:   
                new_content = original.replace(search, replace)
                try:
                    is_error,error=self.check_syntax_error(new_content)
                    if not is_error:
                        logger.info(f"Code edit applied successfully to {file_path}")
                        self._save(file_path, new_content)
                        # Find the position of the replacement and extract context
                        replace_pos = new_content.find(replace)
                        if replace_pos != -1:
                            lines = new_content.split('\n')
                            # Find which line number the replacement starts at
                            chars_so_far = 0
                            replace_line_start = 0
                            for i, line in enumerate(lines):
                                if chars_so_far + len(line) >= replace_pos:
                                    replace_line_start = i
                                    break
                                chars_so_far += len(line) + 1  # +1 for newline
                            
                            # Calculate how many lines the replacement spans
                            replace_lines_count = replace.count('\n') + 1
                            replace_line_end = replace_line_start + replace_lines_count - 1
                            
                            # Extract 20 lines before and after
                            start_line = max(0, replace_line_start - 5)
                            end_line = min(len(lines), replace_line_start + 5)
                            
                            context_lines = []
                            for i in range(start_line, end_line):
                                line_num = i + 1
                                # Mark the edited lines with >>> prefix
                                if replace_line_start <= i <= replace_line_end:
                                    prefix = ">>> "
                                else:
                                    prefix = "    "
                                context_lines.append(f"{prefix}{line_num:4}| {lines[i]}")
                            
                            context = "\n".join(context_lines)
                            return f"ok, code edit applied successfully. Here is the edited code (lines {start_line+1}-{end_line}):\n\n{context}"
                        else:
                            return "ok, code edit applied successfully"
                    else:
                        error.message="code edit failed. "+error.message
                        raise error
                except Exception as e:
                    return f"Error: syntax error in file {file_path}. {str(e)}"
            case num_hits:
                logger.warning(f"Search string found {num_hits} times in {file_path}, cannot apply edit")
                return f"Error: search string found {num_hits} times in file '{file_path}'.\nPlease reformulate your search and replace to apply only one change."
    
    @EnhancedToolManager.tool
    def finish(self):
        '''
        Signals completion of the current workflow execution
        Arguments:
            None
        '''
        logger.info("Finish tool called, generating final patch")
        patch = self.get_final_git_patch()
        patch_content = (patch or "").strip()

        if not patch_content:
            logger.error("Finish called but patch is empty")
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.INVALID_TOOL_CALL.name,
                "Cannot finish: git patch is empty. Please ensure your changes are saved before finishing."
            )

        if patch_content.lower().startswith("error"):
            logger.error(f"Finish called but patch generation failed: {patch_content[:200]}")
            raise EnhancedToolManager.Error(
                EnhancedToolManager.Error.ErrorType.UNKNOWN.name,
                patch_content
            )

        self.latest_generated_patch = patch
        logger.info(f"Finish completed successfully, patch length: {len(patch)}")
        return "finish"

    @EnhancedToolManager.tool
    def finish_test_runner_and_mode(self, test_runner: str, test_runner_mode: str):
        '''
        Signals completion of the test runner and mode finding workflow execution
        Arguments:
            test_runner: The test runner command. (default is pytest) If it is python file, it should be the full path from the root of the repository.
            test_runner_mode: The test runner mode. Either 'FILE' or 'MODULE' (default is FILE)
        '''
        file_paths = [self.test_runner_test_file]
        if test_runner != 'pytest' and test_runner != 'unittest':
            if test_runner_mode == "MODULE":
                _file_paths = [filepath_to_module(f, self.repo_dir, test_runner) for f in file_paths]
            else:
                _file_paths = [clean_filepath(f, self.repo_dir, test_runner) for f in file_paths]
            cmd = f"{test_runner} {' '.join(_file_paths)}"

            logger.info(f"Get Test Runner Mode Command: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            output = (result.stdout or "") + (result.stderr or "")

            logger.info(f"Get Test Runner Mode Output: {output}")
            has_dependency_error = self._check_dependency_errors(output, ["invalid command", "command not found"])
            if not has_dependency_error:
                return test_runner, test_runner_mode
        
        return "pytest", "FILE"

def set_env_for_agent():
    logger.debug("Setting up environment for agent")
    if os.getcwd() not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ.get("PYTHONPATH","")+":"+os.getcwd()
    if Path(os.getcwd()+"/lib").exists() and os.getcwd()+"/lib" not in os.environ.get("PYTHONPATH",""):
        os.environ["PYTHONPATH"]=os.environ["PYTHONPATH"]+":"+os.getcwd()+"/lib"

    work_dir = os.getcwd()
    original_cwd = os.getcwd()
    
    try:
        os.chdir(work_dir)
        if not os.path.exists(".git"):
            logger.info("Initializing git repository")
            subprocess.run(["git", "init"], check=True)
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
            subprocess.run(["git", "config", "--global", "user.email", "agent@sandbox.local"], check=True)
            subprocess.run(["git", "config", "--global", "user.name", "sandbox_agent"], check=True)
            subprocess.run(["git", "add", "."], check=True)
            subprocess.run(["git", "commit", "-m", "Initial commit"], check=False, capture_output=True, text=True)
            logger.info("Git repository initialized successfully")
        else:
            logger.debug("Git repository already exists")
            subprocess.run(["git", "config", "--global", "--add", "safe.directory", work_dir])
    except Exception as e:
        logger.warning(f"Error setting up environment: {e}")
    finally:
        os.chdir(original_cwd)

def clean_code_response(response: str) -> str:
    return response.strip().removeprefix('```python').removeprefix('```').removesuffix('```').strip()

def is_all_tests_passed(output: str) -> bool:
    return not any(k in output.upper() for k in ['FAILED', 'ERROR:', 'FAIL:', 'TRACEBACK'])
    
def extract_and_write_files(initial_solution: str, base_dir: str = ".") -> list:
    import os
    
    logger.debug(f"Extracting and writing files from solution, base_dir={base_dir}")
    if not initial_solution.strip():
        logger.warning("extract_and_write_files called with empty solution")
        return []
    
    created_files = []
    current_file, content = None, []
    
    def write_file():
        if current_file and content:
            path = os.path.join(base_dir, current_file)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content).strip())
            created_files.append(path)
            logger.debug(f"Created file: {path}")

    for line in initial_solution.split('\n'):
        stripped = line.strip()
        if (stripped.endswith('.py') and ' ' not in stripped and 
            len(stripped) > 3 and not stripped.startswith('#')):
            write_file()
            current_file, content = stripped, []
        elif current_file:
            content.append(line)
    
    write_file()
    logger.info(f"Extracted and wrote {len(created_files)} files: {created_files}")
    return created_files

def filepath_to_module(file_path: str, repo_path: str, test_runner: str) -> str:
    """Convert file path to Python module notation."""
    root_path = os.path.abspath(repo_path)
    abs_filepath = os.path.abspath(file_path)
    module_path = os.path.splitext(abs_filepath)[0]
    # Try to strip root_path first
    if module_path.startswith(root_path):
        module_path = module_path[len(root_path):].lstrip(os.path.sep)
    else:
        parent_path = os.path.dirname(root_path)
        if module_path.startswith(parent_path):
            module_path = module_path[len(parent_path):].lstrip(os.path.sep)
    # Try to strip test_runner directory if it exists in the path
    test_runner_dir = os.path.dirname(test_runner)
    if test_runner_dir and module_path.startswith(test_runner_dir):
        module_path = module_path[len(test_runner_dir):].lstrip(os.path.sep)
    
    result = module_path.replace(os.path.sep, '.')
    print("final module: ", result)
    return result
def clean_filepath(file_path: str, repo_path: str, test_runner: str) -> str:
    path = os.path.splitext(os.path.abspath(file_path))[0].removeprefix(os.path.abspath(repo_path)).lstrip(os.path.sep)
    return path.removeprefix(os.path.dirname(test_runner)).lstrip(os.path.sep) if os.path.dirname(test_runner) else path

def get_directory_tree(start_path: str = '.', max_depth: int = 2) -> str:
    # Fallback to Python implementation (works on both Linux and Windows)
    result = []
    
    # Calculate start depth properly using os.path.normpath for cross-platform compatibility
    start_path_abs = os.path.normpath(os.path.abspath(start_path))
    # Split and filter empty strings (handles both / and \ separators)
    start_path_parts = [p for p in start_path_abs.split(os.sep) if p]
    start_depth = len(start_path_parts)

    for root, dirs, files in os.walk(start_path):
        # Calculate current depth using os.path.normpath for cross-platform compatibility
        root_abs = os.path.normpath(os.path.abspath(root))
        # Split and filter empty strings (handles both / and \ separators)
        root_path_parts = [p for p in root_abs.split(os.sep) if p]
        current_depth = len(root_path_parts) - start_depth
        
        # Remove hidden dirs so walk doesn't recurse into them
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Stop recursing if we've reached or exceeded max_depth
        if max_depth is not None and current_depth >= max_depth:
            # Clear dirs to prevent further recursion
            dirs.clear()
        
        # Skip directories beyond max_depth (but still collect files at current depth)
        # Note: We check current_depth > max_depth (not >=) to allow collecting files at max_depth
        if max_depth is not None and current_depth > max_depth:
            continue

        # Collect both directories and files (for better tree structure)
        # Include the root directory itself if it's not the start_path and not hidden
        if root != start_path and not any(part.startswith('.') for part in root.split(os.sep)):
            result.append(root)
        
        # Collect visible files at the current depth (including files in start_path)
        for f in files:
            if not f.startswith('.'):
                file_path = os.path.join(root, f)
                # Normalize path to handle '.' and '..' properly
                file_path = os.path.normpath(file_path)
                result.append(file_path)

    return "\n".join(result)

def determine_temperature(problem_statement: str) -> float:
    def validate_response(response: dict) -> tuple[bool, str]:
        if "temperature" not in response:
            return False, "Required key temperature not found in response"

        temperature = response.get("temperature")

        if temperature is None or not isinstance(temperature, float):
            return False, "Required key temperature not found in response"
        return True, ""

    messages = [
        {"role": "system", "content": TEMPERATURE_DETERMINATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem statement: {problem_statement}"}
    ]
    try:
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0)
        response=response.replace('```json','').strip('```').strip()
        response = json.loads(response)
        
        is_valid, error_msg = validate_response(response)
        if is_valid:
            return response.get("temperature", 0.0)
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "Keep clarifying the temperature until you have a valid float."})
        
    except Exception as e:
        pass
    
    return 0

def check_problem_type(problem_statement: str) -> str:
    PROBLEM_TYPE_CHECK_PROMPT = textwrap.dedent(
        '''
        You are the problem type checker that will categories problem type into:

        1. CREATE: If the problem statement is about creating a new functionality from scratch.
        2. FIX: If the problem statement is about fixing a bug, creating a new functionality or improving the existing codebase.

        Only respond with the "FIX" or "CREATE".
        '''
    )
    
    # Get directory tree with retries until we get valid output
    directory_tree = ""
    tree_retry = 0
    max_tree_retries = 10
    while tree_retry < max_tree_retries:
        try:
            tree_output = get_directory_tree()
            print(tree_output)
            # Check if output is valid (not empty, not an error message)
            if tree_output and tree_output.strip() and not tree_output.strip().lower().startswith("error"):
                directory_tree = tree_output.strip()
                logger.info(f"Successfully retrieved directory tree (attempt {tree_retry + 1})")
                break
            else:
                logger.warning(f"Invalid directory tree output (attempt {tree_retry + 1}): {tree_output[:100] if tree_output else 'empty'}")
                tree_retry += 1
                time.sleep(0.5)  # Brief pause before retry
        except Exception as e:
            logger.error(f"Error getting directory tree (attempt {tree_retry + 1}): {e}")
            tree_retry += 1
            time.sleep(0.5)
    
    user_content = problem_statement

    # If we still don't have valid directory tree after retries, use empty string
    if not directory_tree:
        logger.warning("Failed to get valid directory tree after retries, proceeding without it")
        directory_tree = ""
    else:
        # Truncate directory tree if it exceeds 1000 tokens
        # Estimate tokens: ~4 characters per token (conservative estimate)
        MAX_DIRECTORY_TREE_TOKENS = 1000
        estimated_tokens = len(directory_tree) // 4
        
        if estimated_tokens > MAX_DIRECTORY_TREE_TOKENS:
            # Calculate max characters for 1000 tokens (leave some buffer)
            max_chars = MAX_DIRECTORY_TREE_TOKENS * 4 - 100  # Reserve 100 chars for truncation marker
            if len(directory_tree) > max_chars:
                directory_tree = directory_tree[:max_chars] + "\n... [directory tree truncated]"
                logger.info(f"Directory tree truncated from ~{estimated_tokens} tokens to ~{MAX_DIRECTORY_TREE_TOKENS} tokens")
        
        user_content += f"\n# Project Tree Structure: \n{directory_tree}"
    
    # Now proceed with problem type check using the directory tree
    # This is critical - keep retrying until we get a valid response
    # If this fails, all other processes will fail
    retry = 0
    response = None
    while True:
        try:
            messages = [
                {"role": "system", "content": PROBLEM_TYPE_CHECK_PROMPT},
                {"role": "user", "content": user_content}
            ]
            response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME)
            
            # Check if response is valid
            if response and response.strip() and response.strip() in [PROBLEM_TYPE_CREATE, PROBLEM_TYPE_FIX]:
                valid_response = response.strip()
                logger.info(f"Successfully determined problem type: {valid_response} (after {retry + 1} attempt(s))")
                return valid_response
            else:
                retry += 1
                logger.warning(f"Invalid problem type response (attempt {retry}): {response[:100] if response else 'empty'}. Retrying...")
                
        except Exception as e:
            retry += 1
            logger.error(f"Error checking problem type (attempt {retry}): {e}. Retrying...")
        
        # Exponential backoff: start with 2 seconds, increase gradually
        sleep_time = min(2 * (1.2 ** retry), 30)  # Cap at 30 seconds
        time.sleep(sleep_time)
        
        # Log progress every 10 attempts to show it's still working
        if retry % 10 == 0:
            logger.warning(f"Still retrying problem type check (attempt {retry}). This is critical - will not give up.")

def generate_initial_solution(problem_statement: str, code_skeleton: str, temperature: float = 0.7) -> str:
    GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT = textwrap.dedent(
        """
        You are an expert Python developer. Your task is to generate a complete, working Python solution for the given problem statement.
        
        Strict Requirements:
        1. Output the full content of Python files along with their file names. You **MUST** output the **file name** along with file content.
        2. Do not include explanations, comments, or markdown formatting in the main code.
        3. Use only standard Python (no external libraries).
        4. Implement all required classes and functions exactly with the same names as in the initial code stub.
        5. You may add helper functions or classes if needed, but do not remove or rename the original ones.
        6. Ensure the solution handles all edge cases, validates inputs, and produces correct outputs.
        7. The solution must be executable as-is with no placeholders or TODOs.
        8. If problem statement doesn't explicitely requires a list of strings as a response, do not use list of strings for multiline text problems, just use raw string format.
        9. **IMPORTANT**: Add clear comments above each edge case handling section to identify which specific edge case is being addressed. Use the format: `# Edge Case: [description of the edge case]`
        10. **IMPORTANT**: Add a comment at the end of each function/class that lists all edge cases handled, using the format: `# Handled Edge Cases: [list of edge cases]`

        Return only the final Python code.
        
        Response Examples:
        ```python
        a.py
        {{content}}
        
        b.py
        {{content}}
        ```
        """
    )
    INFINITE_LOOP_CHECK_PROMPT = textwrap.dedent(
        """
        You are an expert code reviewer specializing in infinite loop detection and prevention. Your task is to analyze the generated Python code for potential infinite loops and provide a corrected version if issues are found.

        CRITICAL INFINITE LOOP DETECTION:
        1. Check for while True: loops without guaranteed exit conditions
        2. Verify all while loops have clear termination conditions
        3. Ensure recursive functions have proper base cases
        4. Look for loops that depend on external state that might never change
        5. Check for patterns that could lead to infinite iteration

        If you find potential infinite loops:
        - Provide a corrected version of the code
        - Ensure all loops have finite termination conditions
        - Add reasonable iteration limits or timeout mechanisms where appropriate

        If no infinite loops are detected:
        - Return the original code unchanged

        STRICT REQUIREMENT: Return the final Python code along with file names. Do not include any explanations, comments, or additional text.

        example:
        ```python
        a.py
        contents of a.py

        b.py
        contents of b.py
        ```
        """
    )
    retry = 0
    code_generation_messages = [
        {"role": "system", "content": GENERATE_SOLUTION_WITH_MULTI_STEP_REASONING_PROMPT},
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nInitial python files:\n{code_skeleton}\nGenerate the complete and correct implementation in python files.\n\nSTRICT REQUIREMENT: - You **MUST** output the **file name** along with file content.\nexample:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
        },
    ]
    logger.info(f"Generating initial solution with temperature={temperature}")
    while retry < 10:
        try:
            logger.debug(f"Code generation attempt {retry + 1}/10")
            code_response = EnhancedNetwork.make_request(code_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature)
            loop_check_messages = [
                {"role": "system", "content": INFINITE_LOOP_CHECK_PROMPT},
                {
                    "role": "user",
                    "content": f"Generated Code:\n{code_response}\n\nAnalyze this code for potential infinite loops and provide a corrected version if any issues are found. Return ONLY the final Python code.",
                },
            ]
            loop_check_response = EnhancedNetwork.make_request(loop_check_messages, model=QWEN_MODEL_NAME)
            # Clean up the final response (use compat response as it's the final validated version)
            solution = clean_code_response(loop_check_response)
            lines = solution.split("\n")
            if lines[0].endswith(".py") == False:
                logger.warning(f"Generated solution doesn't start with .py file name, retrying (attempt {retry + 1})")
                retry += 1
                code_generation_messages.append({"role": "assistant", "content": loop_check_response})
                code_generation_messages.append(
                    {
                        "role": "user",
                        "content": f"Include file name in the response. example:\n```python\na.py\ncontents of a.py\n\nb.py\ncontents of b.py\n```",
                    }
                )
                continue
            logger.info("Successfully generated initial solution")
            return solution
        except Exception as e:
            logger.warning(f"Error generating initial solution (attempt {retry + 1}): {e}")
            retry += 1
            time.sleep(2)
    if retry >= 10:
        logger.error("Failed to generate initial solution after 10 attempts")
        return ""
    return ""

def generate_single_testset(problem_statement: str, files_to_test: str, code_skeleton: str, temperature: float = 0.0) -> tuple[str, set]:
    """Generate a single test set and return (testcode, function_names)"""
    
    GENERATE_TESTCASES_PROMPT = textwrap.dedent(
        """
        You are an expert Python unittest testcase developer. 
            Important points:-
            - you have generation limit of 2048 tokens. Hence you must stop generating more test cases when you are near the limit.
            - If you get syntax error, check if last assistant response was truncated. If yes, then skip last couple of test cases to fit in.
            
            You must respond directly with the test cases in the following format. 
            =========TEST_CASES
            <<test cases>>
            Do not include anything else. For Example:
            =========TEST_CASES

            import unittest
            from main_module import (
                main_func
            )

            class TestFuncA(unittest.TestCase):
                def test_main_func(self):
                    self.assertEqual(main_func(), "expected_output")

            if __name__ == "__main__":
                unittest.main()
        """
    )

    retry = 0
    test_generation_messages = [
        {
            "role": "system",
            "content": GENERATE_TESTCASES_PROMPT
        },
        {
            "role": "user",
            "content": f"Problem Statement:\n{problem_statement}\n\nFiles To Test: {files_to_test}\n\nCode skeleton: \n{code_skeleton}\n\nGenerate the complete and correct testcases in python files.\n\nSTRICT REQUIREMENT: You **MUST** output the **file name** along with file content.\nexample:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"
        }
    ]
    
    logger.debug(f"Generating test set with temperature={temperature}")
    while retry < 10:
        try:
            logger.debug(f"Test generation attempt {retry + 1}/10")
            testcode_response = EnhancedNetwork.make_request(test_generation_messages, model=QWEN_MODEL_NAME, temperature=temperature)

            testcases = clean_code_response(testcode_response)
            
            lines = testcases.split("\n")
            if lines[0].endswith(".py") == False:
                logger.warning(f"Generated test doesn't start with .py file name, retrying (attempt {retry + 1})")
                retry += 1
                test_generation_messages.append({"role": "assistant", "content": testcode_response})
                test_generation_messages.append({"role": "user", "content": f"Include file name in the response. example:\n```python\ntest_a.py\ncontents of test_a.py\n\ntest_b.py\ncontents of test_b.py\n```"})

                continue

            logger.info("Successfully generated test set")
            return testcases
            
        except Exception as e:
            logger.warning(f"Error generating test set (attempt {retry + 1}): {e}")
            retry += 1

            time.sleep(2)
    
    logger.error("Failed to generate test set after 10 attempts")
    return "", set()

def advanced_approach(code_skeleton: str, problem_statement: str, timeout: int) -> str:
    logger.info("Starting advanced_approach")
    tool_manager = EnhancedToolManager()
    temperature = determine_temperature(problem_statement)
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, temperature)
    if not initial_solution:
        logger.error("Failed to generate initial solution in advanced_approach")
        return ""
    created_files = extract_and_write_files(initial_solution)
    logger.info(f"Created {len(created_files)} files from initial solution")
    
    patch = tool_manager.get_final_git_patch()
    logger.info(f"Generated patch with length {len(patch)}")
    return patch

def basic_approach(code_skeleton: str, problem_statement: str, temperature: float = 0.0) -> str:  
    logger.info(f"Starting basic_approach with temperature={temperature}")
    initial_solution = generate_initial_solution(problem_statement, code_skeleton, temperature)
    if not initial_solution:
        logger.error("Failed to generate initial solution in basic_approach")
        return None
    created_files = extract_and_write_files(initial_solution)
    logger.info(f"Created {len(created_files)} files from initial solution")

    test_cases = generate_single_testset(problem_statement, created_files, code_skeleton, temperature)
    if not test_cases:
        logger.error("Failed to generate test cases in basic_approach")
        return None
    test_files = extract_and_write_files(test_cases)
    logger.info(f"Created {len(test_files)} test files")

    for file in test_files:
        logger.debug(f"Running tests in {file}")
        result = subprocess.run(["python", file], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)

        if not is_all_tests_passed(result.stdout):
            logger.warning(f"Tests failed in {file}")
            return None

    logger.info("All tests passed in basic_approach")
    return initial_solution

def generate_fallback_solution(problem_statement: str, cot: EnhancedCOT, tool_manager, timeout_remaining: int = 60) -> str:
    """
    Generate a fallback solution when the main workflow times out or fails.
    Uses problem statement + agent's exploration history to generate a best-effort patch.
    
    Args:
        problem_statement: Original problem description
        cot: Chain of thought from the agent's execution
        tool_manager: Tool manager instance for file operations
        timeout_remaining: Time budget for fallback generation
        
    Returns:
        Git patch string (may be empty if generation fails)
    """
    logger.info("=" * 80)
    logger.info("GENERATING FALLBACK SOLUTION - Main workflow did not complete")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    explored_files = set()
    relevant_content = {}
    search_patterns = []
    
    for thought in cot.thoughts:
        if thought.is_deleted:
            continue
            
        if thought.next_tool_name in ["get_file_content", "search_in_specified_file_v2", "apply_code_edit"]:
            if isinstance(thought.next_tool_args, dict) and "file_path" in thought.next_tool_args:
                file_path = thought.next_tool_args["file_path"]
                explored_files.add(file_path)
        
        # Track search patterns to understand what agent was looking for
        if thought.next_tool_name == "search_in_all_files_content":
            if isinstance(thought.next_tool_args, dict) and "search_term" in thought.next_tool_args:
                search_patterns.append(thought.next_tool_args["search_term"])
    
    logger.info(f"Agent explored {len(explored_files)} files: {explored_files}")
    logger.info(f"Agent searched for patterns: {search_patterns[:5]}")
    
    # Get content of explored files (limit to most relevant)
    for file_path in list(explored_files)[:10]:
        try:
            if os.path.exists(file_path) and file_path.endswith('.py'):
                content = tool_manager._get_file_content(file_path, limit=500)
                relevant_content[file_path] = content
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
    
    # Find test files
    test_files = [f for f in explored_files if "test" in f.lower()]
    if not test_files:
        # Look for test files in current directory
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    test_path = os.path.join(root, file)
                    test_files.append(test_path)
                    if len(test_files) >= 3:
                        break
            if len(test_files) >= 3:
                break
    
    logger.info(f"Identified test files: {test_files}")
    
    # Build context for LLM
    context_parts = []
    for file_path, content in list(relevant_content.items())[:8]:
        context_parts.append(f"=== {file_path} ===\n{content}\n")
    
    context = "\n".join(context_parts)
    
    fallback_prompt = textwrap.dedent(f"""
    You are an expert code repair assistant. An automated agent attempted to fix this issue but ran out of time.
    Based on the problem statement and the code that was explored, generate the minimal fix needed.
    
    # Problem Statement
    {problem_statement}
    
    # Files Explored by Agent
    {', '.join(explored_files) if explored_files else 'None - agent timed out early'}
    
    # Search Patterns Agent Used
    {', '.join(search_patterns[:5]) if search_patterns else 'None'}
    
    # Relevant Code Context
    {context if context else 'No code context available'}
    
    # Test Files
    {', '.join(test_files) if test_files else 'No test files identified'}
    
    # Instructions
    Generate a complete, minimal solution. For each file that needs to be modified:
    
    1. Specify the file path
    2. Show the EXACT code to find (must match exactly, including indentation)
    3. Show the EXACT replacement code
    
    Use this format for EACH change:
    
    FILE: path/to/file.py
    SEARCH:
    ```python
    exact code to find and replace (with proper indentation)
    ```
    REPLACE:
    ```python
    new code (with proper indentation)
    ```
    
    Important:
    - Make minimal changes that directly address the problem
    - Preserve existing indentation and formatting
    - Only modify what's necessary
    - Ensure the SEARCH block matches existing code exactly
    - If you're unsure about a file, skip it rather than guessing
    
    Generate the solution now:
    """)
    
    try:
        logger.info("Requesting fallback solution from LLM...")
        messages = [
            {"role": "system", "content": "You are a code repair expert. Generate minimal, precise code fixes in the specified format."},
            {"role": "user", "content": fallback_prompt}
        ]
        
        response = EnhancedNetwork.make_request(messages, model=QWEN_MODEL_NAME, temperature=0.3)
        logger.info(f"Received fallback solution response ({len(response)} chars)")
        
        # Log the response for debugging
        logger.debug(f"Fallback LLM response:\n{response[:1000]}...")
        
        # Parse and apply the changes
        edits_applied = apply_fallback_changes(response, tool_manager)
        
        if edits_applied > 0:
            # Generate final patch
            patch = tool_manager.get_final_git_patch()
            logger.info(f"✓ Fallback solution generated patch of length: {len(patch)} ({edits_applied} edits applied)")
            return patch
        else:
            logger.warning("Fallback solution: No edits could be applied")
            return ""
        
    except Exception as e:
        logger.error(f"Fallback solution generation failed: {e}")
        logger.error(traceback.format_exc())
        return ""
    finally:
        elapsed = time.time() - start_time
        logger.info(f"Fallback generation took {elapsed:.2f} seconds")

def apply_fallback_changes(llm_response: str, tool_manager) -> int:
    """
    Parse LLM response containing FILE/SEARCH/REPLACE blocks and apply changes.
    Returns the number of successful edits applied.
    """
    edits_applied = 0
    
    file_pattern = r'FILE:\s*([^\n]+)'
    search_pattern = r'SEARCH:\s*```(?:python)?\s*\n(.*?)```'
    replace_pattern = r'REPLACE:\s*```(?:python)?\s*\n(.*?)```'
    
    file_matches = list(re.finditer(file_pattern, llm_response, re.IGNORECASE))
    
    for i, file_match in enumerate(file_matches):
        file_path = file_match.group(1).strip()
        
        start_pos = file_match.end()
        end_pos = file_matches[i + 1].start() if i + 1 < len(file_matches) else len(llm_response)
        block_text = llm_response[start_pos:end_pos]
        
        # Extract SEARCH and REPLACE from this block
        search_match = re.search(search_pattern, block_text, re.DOTALL)
        replace_match = re.search(replace_pattern, block_text, re.DOTALL)
        
        if search_match and replace_match:
            search_code = search_match.group(1).strip()
            replace_code = replace_match.group(1).strip()
            
            try:
                logger.info(f"Applying fallback edit to {file_path}")
                
                if not os.path.exists(file_path):
                    logger.warning(f"File {file_path} not found, skipping")
                    continue
                
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if search_code in content:
                    new_content = content.replace(search_code, replace_code, 1)
                    
                    # Check for syntax errors
                    is_error, error = tool_manager.check_syntax_error(new_content, file_path)
                    if not is_error:
                        with open(file_path, 'w') as f:
                            f.write(new_content)
                        edits_applied += 1
                        logger.info(f"Successfully applied edit to {file_path}")
                    else:
                        logger.warning(f"Syntax error in fallback edit for {file_path}: {error}")
                else:
                    logger.warning(f"Search string not found in {file_path}")
                    
            except Exception as e:
                logger.error(f"Error applying fallback edit to {file_path}: {e}")
                continue
    
    logger.info(f"Applied {edits_applied} fallback edits")
    return edits_applied

def execute_agent_workflow(
    system_prompt: str,
    instance_prompt: str,
    timeout: int, 
    run_id_1: str,
    cot: EnhancedCOT,
    tool_manager: EnhancedToolManager,
    n_max_steps: int = MAX_FIX_TASK_STEPS,
    log_prefix: str = "agent",
    finish_tool_name: str = "finish",
    max_obeservation_tokens: int = 50000,
    max_raw_observation_tokens: int = MAX_TOOL_CALL_RESPONSE_TOKENS,
    model_name: str = GLM_MODEL_NAME
):
    global run_id, ESTIMATED_INPUT_COST, ESTIMATED_OUTPUT_COST
    run_id=run_id_1
    logger.info(f"[{log_prefix}]Starting execute_agent_workflow with run_id={run_id_1}, timeout={timeout}, max_steps={n_max_steps}")

    start_time = time.time()
    last_test_result = 'success'
    logs: List[str] = []
    
    # Initialize variables to avoid UnboundLocalError
    raw_text = ""
    total_attempts = 0
    error_counter = {}
    next_thought = None
    next_tool_name = None
    next_tool_args = None
    
    for step in range(n_max_steps):
        elapsed_time = time.time() - start_time
        logger.info("="*100)
        logger.info(f"[{log_prefix}] Execution step {step + 1}/{n_max_steps}, Elapsed time: {time.time() - start_time} seconds, timeout: {timeout} seconds, Number of thoughts in COT: {len(cot.thoughts)}, Estimated cost: {ESTIMATED_INPUT_COST + ESTIMATED_OUTPUT_COST}/(Input: {ESTIMATED_INPUT_COST}, Output: {ESTIMATED_OUTPUT_COST})")

        if time.time() - start_time > timeout:
            logger.warning(f"[{log_prefix}] Workflow timeout reached: {elapsed_time:.2f}s > {timeout}s")
            cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break
        
        if ESTIMATED_INPUT_COST + ESTIMATED_OUTPUT_COST > TOTAL_COST_THRESHOLD:
            logger.warning(f"[{log_prefix}] Estimated cost is too high: {ESTIMATED_INPUT_COST + ESTIMATED_OUTPUT_COST} > {TOTAL_COST_THRESHOLD}")
            cot.add_action(EnhancedCOT.Action(next_thought="estimated cost is too high",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))
            break

        messages: List[Dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instance_prompt},
            ]
        
        messages.extend(cot.to_str())

        temperature = 0
        selected_model = model_name
        if cot.is_thought_repeated():
            logger.warning(f"[{log_prefix}] Thought repeated {cot.repeated_thoughts} times at step {step + 1}")
            last_thought = cot.thoughts[-1]
            messages.append({"role": "user", "content": DO_NOT_REPEAT_TOOL_CALLS.format(previous_response=f"next_tool_name:{last_thought.next_tool_name}\n next_tool_args:{last_thought.next_tool_args}")})

            if cot.repeated_thoughts > 1:
                temperature = min(cot.repeated_thoughts / 10, 0.7)
                selected_model = AGENT_MODELS[random.randint(0, len(AGENT_MODELS)-1)] if cot.repeated_thoughts > 2 else model_name
                logger.info(f"[{log_prefix}] Adjusted temperature to {temperature} and model to {selected_model} due to repeated thoughts")

        try:
            logger.debug(f"[{log_prefix}] Calling inference with model {selected_model}, temperature {temperature}")
            inference_start_time = time.time()
            next_thought, next_tool_name, next_tool_args,raw_text,total_attempts,error_counter,messages = EnhancedNetwork.inference(messages, model=selected_model, run_id=run_id, temperature=temperature)
            logger.info(f"[{log_prefix}] next_thought: {next_thought}\nnext_tool_name: {next_tool_name}\nnext_tool_args: {next_tool_args}\nmodel: {selected_model}\nmodel inference time: {time.time() - inference_start_time} seconds")
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"\n\nERROR: {repr(e)} {traceback.format_exc()}"
            logger.error(f"[{log_prefix}]Inference failed at step {step + 1}: {error_msg}")

            cot.add_action(EnhancedCOT.Action(next_thought=error_msg,next_tool_name="",next_tool_args={},observation="",is_error=True,raw_response=raw_text or "",total_attempts=total_attempts,inference_error_counter=error_counter or {},request_data=messages))
            break

        try:
            if isinstance(next_tool_name, str) and ('"' in next_tool_name or "'" in next_tool_name):
                next_tool_name=next_tool_name.replace('"','')
                next_tool_name=next_tool_name.replace("'","")
            
            logger.info(f"[{log_prefix}] Executing tool: {next_tool_name} with args: {next_tool_args}")
            next_observation = tool_manager.get_tool(next_tool_name)(**next_tool_args) if next_tool_args else tool_manager.get_tool(next_tool_name)()
            logger.info(f"[{log_prefix}] next_observation:\n {next_observation}")
            # logger.debug(f"Tool {next_tool_name} executed successfully, observation length: {len(str(next_observation))}")

            # Check if input token size exceeds 3000 tokens and save to file if needed
            estimated_tokens = Utils.count_tokens([{'content': next_observation}])
            logger.info(f"[{log_prefix}] Estimated observation tokens: {estimated_tokens}")

            if estimated_tokens > max_obeservation_tokens:
                error_msg = f"Error: Tool output from '{next_tool_name}' exceeded token limit ({estimated_tokens} tokens > 50000 tokens limit). The response is too large to process. Please use more specific queries, target smaller file ranges, or break the request into smaller operations."
                cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
                logger.warning(f"[{log_prefix}] Tool output exceeded token limit: {estimated_tokens} tokens for tool '{next_tool_name}'")
                continue
            
            elif estimated_tokens > max_raw_observation_tokens:
                # Save the large observation to a file
                observation_path = tool_manager._save_large_observation(str(next_observation), next_tool_name)
                observation_msg = f"Observation result exceeded token limit ({estimated_tokens} tokens > {max_raw_observation_tokens} tokens limit). The full output has been saved to: {observation_path}. You can read this file using the get_file_content or search_in_specified_file_v2 tool if needed."
                cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=observation_msg,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
                logger.warning(f"[{log_prefix}] Tool output exceeded token limit: {estimated_tokens} tokens for tool '{next_tool_name}', saved to {observation_path}")
                continue

            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=next_observation,is_error=False,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            if next_tool_name == 'run_repo_tests':
                if 'failed' in str(next_observation).lower():
                    last_test_result = 'failed'
                    logger.warning(f"[{log_prefix}] Tests failed")
                else:
                    last_test_result = 'success'
                    logger.info(f"[{log_prefix}] Tests passed")

        except EnhancedToolManager.Error as e:
            import traceback  # Ensure traceback is accessible
            error_msg=f"observation: {e.message}"
            logger.error(f"[{log_prefix}] Tool error for {next_tool_name}: {e.message}")

            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        except Exception as e:
            import traceback  # Ensure traceback is accessible
            error_traceback=traceback.format_exc()
            if isinstance(e,TypeError):
                error_msg=f"observation: {str(e)}"
            else:
                error_msg=f"observation: {repr(e)} {error_traceback}"
            logger.error(f"[{log_prefix}] Unexpected error executing tool {next_tool_name}: {error_msg}")

            cot.add_action(EnhancedCOT.Action(next_thought=next_thought,next_tool_name=next_tool_name,next_tool_args=next_tool_args,observation=error_msg,is_error=True,raw_response=raw_text,total_attempts=total_attempts,inference_error_counter=error_counter,request_data=messages))
            continue
        
        if next_tool_name == finish_tool_name:
            if next_tool_name == "finish":
                logger.info("Finish tool called")
                # if last_test_result == 'failed':
                #     logger.warning("[{log_prefix}] Finish called but tests failed, continuing workflow")
                #     messages.append({"role": "user", "content": "The tests failed. Please fix the code and try again."})
                #     continue
                    
                if not tool_manager.has_pending_patch_changes():
                    logger.warning("[{log_prefix}] Finish called but no code changes detected, rejecting finish")
                    rejection_msg = "Finish request rejected: no code changes detected in the repository. Re-evaluate the problem or provide justification."
                    if cot.thoughts:
                        cot.thoughts[-1].observation = rejection_msg
                        cot.thoughts[-1].is_error = True
                    continue

                logger.info("Finish tool called successfully, ending workflow")
                break
            
            if next_tool_name == "finish_test_runner_and_mode":
                return next_observation

    else:
        logger.warning(f"[{log_prefix}] Workflow completed after reaching max_steps ({n_max_steps})")
        cot.add_action(EnhancedCOT.Action(next_thought="global timeout reached",next_tool_name="",next_tool_args={},observation="",is_error=True,inference_error_counter={},request_data=[]))

        # if n_max_steps < MAX_FIX_TASK_STEPS:
        #     logger.warning(f"Max steps ({n_max_steps}) is less than MAX_FIX_TASK_STEPS ({MAX_FIX_TASK_STEPS}), returning None")
        #     return None, cot

    if finish_tool_name == "finish":
        logger.info("Generating final git patch")
        patch = tool_manager.get_final_git_patch()
        logger.info(f"Workflow completed successfully, patch length: {len(patch)}")
        return patch, cot
    
    if finish_tool_name == "finish_test_runner_and_mode":
        return 'pytest', 'FILE'
    
def find_test_runner_and_mode() -> tuple[str, str]:
    def count_test_cases(file_path: str) -> int:
        """Count the number of test cases (functions starting with 'test_') in a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            import re
            test_functions = re.findall(r'^\s*def\s+test_\w+', content, re.MULTILINE)
            return len(test_functions)
        
        except (FileNotFoundError, UnicodeDecodeError):
            return 0

    global run_id
    test_files = []  # Initialize the test_files list
    test_file_path = None
    
    for root, _, files in os.walk('.'):
        for file in files:
            if 'test_' in file and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    test_files.sort(key=len)

    for path in test_files:
        if count_test_cases(path) > 5:
            test_file_path = path
            break

    if not test_file_path:
        print(f"no test file found")
        return "pytest", "FILE"

    cot = EnhancedCOT(latest_observations_to_keep=10, summarize_batch_size=10)

    tool_manager = FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "list_directory",
            "finish_test_runner_and_mode"
        ],
        test_runner_test_file=test_file_path,
        repo_dir="."
    )

    FIND_TEST_RUNNER_SYSTEM_PROMPT = textwrap.dedent(
        """
        You are a helpful assistant tasked with figuring out how to run the test for the given repository. You have to figure out test runner command and test runner mode.
        
        Test runner mode can be one of the following. No other texts are allowed.
        - MODULE: When the test runner requires a module path to run the test.
        - FILE: When the test runner requires a file path to run the test.

        Follow the below steps to figure out how to run the test:
        - Check the repository files to find out instruction documentations that tells you how to run the test.
        - If you can't find any instruction documentation, finsh the task with default test runner('pytest') and mode('FILE').
        - If you find any instruction documentation contains how to run the test, extract the test run command first. (e.g. test_runner.py test_file)
        - After extracting the test run command, determine the test runner mode by checking the test runner file.
        - Finish the task with `finish_test_runner_and_mode` tool.

        You have access to the following tools:-
        {tools_docs}

        {format_prompt}
        """
    ).format(tools_docs=tool_manager.get_tool_docs(),format_prompt=FORMAT_PROMPT_V0)
    
    try:
        result = execute_agent_workflow(
            system_prompt=FIND_TEST_RUNNER_SYSTEM_PROMPT,
            instance_prompt="Find the test runner and test runner mode for the given repository",
            n_max_steps=10,
            timeout=300,
            run_id_1=run_id,
            finish_tool_name="finish_test_runner_and_mode",
            log_prefix="FIND_TEST_RUNNER_AND_MODE",
            tool_manager=tool_manager,
            cot=cot,
            model_name=KIMI_MODEL_NAME,  
            max_raw_observation_tokens=40000
        )
        logger.info(f"FIND_TEST_RUNNER_AND_MODE result: {result}")
        if result is None or result == "":
            return "pytest", "FILE"
        else:
            return result[0], result[1]
    except Exception as e:
        logger.error(f"Error finding test runner and mode: {e}")
        return "pytest", "FILE"

def fix_task_solve_workflow(problem_statement: str, *, timeout: int, run_id_1: str, test_runner: str, test_runner_mode: str, initial_checkpoint=None) -> tuple[str, EnhancedCOT]:
    
    global run_id, ESTIMATED_INPUT_COST, ESTIMATED_OUTPUT_COST
    run_id=run_id_1
    cot=EnhancedCOT(latest_observations_to_keep=LATEST_OBSERVATIONS_TO_KEEP, summarize_batch_size=SUMMARIZE_BATCH_SIZE)

    tool_manager=FixTaskEnhancedToolManager(
        available_tools=[
            "get_file_content",
            "list_directory",
            "save_file",
            "search_in_all_files_content",
            "search_in_specified_file_v2",
            "run_repo_tests",
            # "run_code",
            "apply_code_edit",
            "generate_tests",
            # "trace_execution",
            # "identify_boundary_conditions",
            "finish"
        ],
        initial_checkpoint=initial_checkpoint,
        test_runner=test_runner,
        test_runner_mode=test_runner_mode
    )
    logger.info(f"Initialized tool manager with {len(tool_manager.TOOL_LIST)} tools")

    system_prompt = FIX_TASK_SYSTEM_PROMPT.format(tools_docs=tool_manager.get_tool_docs(),problem_statement=problem_statement,format_prompt=FORMAT_PROMPT_V0)

    instance_prompt = FIX_TASK_INSTANCE_PROMPT_TEMPLATE
    
    return execute_agent_workflow(
        system_prompt=system_prompt,
        instance_prompt=instance_prompt,
        timeout=timeout,
        run_id_1=run_id,
        cot=cot,
        tool_manager=tool_manager,
        n_max_steps=100,
        log_prefix="MAIN_AGENT",
        model_name=GLM_MODEL_NAME_P
    )

def process_create_task(input_dict):
    def get_code_skeleton() -> str:
        return "\n\n".join(
            f"{f}\n{{\n{open(os.path.join(r, f)).read()}\n}}"
            for r, _, files in os.walk(".")
            for f in files if f.endswith(".py")
        )

    problem_statement = input_dict.get("problem_statement", "")

    tool_manager = EnhancedToolManager()
    code_skeleton = get_code_skeleton()

    timeout = DEFAULT_TIMEOUT

    BASIC_APPROACH_RETRY = 10
    min_temperature = 0.1
    max_temperature = 1.2
    temperature_schedule = []

    # Advanced temperature strategy:
    # - Start low, ramp to mid, jitter at plateau, ramp to high
    warmup_steps = 2
    plateau_steps = 4
    high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    if high_steps < 0:
        warmup_steps = 1
        plateau_steps = BASIC_APPROACH_RETRY // 2
        high_steps = BASIC_APPROACH_RETRY - (warmup_steps + plateau_steps)
    # Warmup phase: gradual increase from min_temperature to mid_temperature
    mid_temperature = 0.6
    for i in range(warmup_steps):
        t = min_temperature + (mid_temperature - min_temperature) * (i / max(1, warmup_steps-1))
        temperature_schedule.append(round(t, 3))
    # Plateau phase: random jitter around mid_temperature
    for _ in range(plateau_steps):
        t = mid_temperature + random.uniform(-0.09, 0.09)
        t = max(min_temperature, min(max_temperature, t))
        temperature_schedule.append(round(t, 3))
    # High phase: progressive ramp to max_temperature
    for i in range(high_steps):
        if high_steps > 1:
            t = mid_temperature + (max_temperature - mid_temperature) * (i / (high_steps-1))
        else:
            t = max_temperature
        temperature_schedule.append(round(t, 3))

    logger.info(f"Starting process_create_task with {BASIC_APPROACH_RETRY} retries, temperature schedule: {temperature_schedule}")
    for attempt, temperature in enumerate(temperature_schedule):
        logger.info(f"Create task attempt {attempt + 1}/{BASIC_APPROACH_RETRY} with temperature={temperature}")
        os.system("git reset --hard")

        initial_solution = basic_approach(
            code_skeleton, problem_statement, temperature=temperature
        )
        if initial_solution is not None:
            logger.info(f"Basic approach succeeded on attempt {attempt + 1}")
            extract_and_write_files(initial_solution)
            patch = tool_manager.get_final_git_patch()
            logger.info(f"Generated patch with length {len(patch)}")
            return patch
        # Adaptive sleep: sleep more for higher temperature to allow model to "think"
        sleep_time = 1 + 0.5 * (temperature - min_temperature)
        logger.debug(f"Basic approach failed, sleeping {sleep_time:.2f}s before next attempt")
        time.sleep(sleep_time)

    logger.warning("Basic approach failed after all retries, falling back to advanced_approach")
    return advanced_approach(code_skeleton, problem_statement, timeout)

def process_fix_task(input_dict: Dict[str, Any]):
    global run_id
    logger.info("Starting process_fix_task")
    # setting environment to include current working directory and lib directory
    problem_text = input_dict.get("problem_statement")
    if not problem_text:
        logger.error("input_dict missing 'problem_statement'")
        raise ValueError("input_dict must contain 'problem_statement'.")
    timeout = int(os.getenv("AGENT_TIMEOUT", str(DEFAULT_TIMEOUT)))
    logger.info(f"Process fix task timeout: {timeout}s")
    
    logs = []
    patch_text = ""  # Initialize to avoid UnboundLocalError
    cot = None  # Track COT for fallback
    
    repo_path = os.getenv("REPO_PATH", "/sandbox/repo")
    repod_dir = repo_path.split('/')[-1]
    if os.path.exists(repod_dir):
        logger.info(f"Changing directory to {repod_dir}")
        os.chdir(repod_dir)

    set_env_for_agent()
    cwd = os.getcwd()
    logger.info(f"Working directory: {cwd}")

    try:
        test_runner, test_runner_mode = find_test_runner_and_mode()
        logger.info(f"test_runner: {test_runner}, test_runner_mode: {test_runner_mode}")

        patch_text, cot = fix_task_solve_workflow(
            problem_text,
            timeout=timeout,
            run_id_1=run_id,
            test_runner=test_runner,
            test_runner_mode=test_runner_mode
        )
        
        # Check if we need fallback solution
        if not patch_text or patch_text is None:
            logger.warning("Main workflow returned no patch, generating fallback solution...")
            
            # Only attempt fallback if we have a valid COT with some exploration history
            if cot and len(cot.thoughts) > 0:
                # Create a tool manager for fallback
                fallback_tool_manager = FixTaskEnhancedToolManager()
                
                patch_text = generate_fallback_solution(
                    problem_statement=problem_text,
                    cot=cot,
                    tool_manager=fallback_tool_manager,
                    timeout_remaining=60
                )
                
                if patch_text:
                    logger.info("✓ Fallback solution generated successfully")
                else:
                    logger.warning("✗ Fallback solution failed to generate patch")
                    patch_text = ""  # Ensure we don't return None
            else:
                logger.warning("No COT history available for fallback generation")
                patch_text = ""  # Ensure we don't return None
        
        logger.info(f"Fix task workflow completed, patch length: {len(patch_text)}")
        logger.info(f"Cot:\n\n {json.dumps(cot.to_str(), indent=4)}")
        os.system("git reset --hard")

    except Exception as e:
        import traceback  # Ensure traceback is accessible
        error_info = f"Error: {e}, {traceback.format_exc()}"
        logger.error(f"Exception in process_fix_task: {error_info}")
    finally:
        os.chdir(cwd)
    return patch_text

def agent_main(input_dict: Dict[str, Any], repo_dir: str = "repo"):
    global DEFAULT_PROXY_URL, DEFAULT_TIMEOUT, run_id
    logger.info(f"Starting agent_main with repo_dir={repo_dir}")
    run_id = os.getenv("RUN_ID", "")
    logger.info(f"Run ID: {run_id}")
    repo_dir = os.path.abspath(repo_dir)
    sys.path.insert(0, repo_dir)
    if os.path.exists(repo_dir):
        logger.info(f"Changing directory to {repo_dir}")
        os.chdir(repo_dir)
    set_env_for_agent()
    
    # Print Python version and installed libraries
    logger.info("="*80)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")
    logger.info("="*80)
    logger.info("Installed packages:")
    try:
        problem_statement = input_dict.get("problem_statement", "")
        logger.info("Checking problem type")
        problem_type = check_problem_type(problem_statement)
        logger.info(f"Problem type determined: {problem_type}")
        if problem_type == PROBLEM_TYPE_FIX:
            logger.info("Processing as FIX task")
            result = process_fix_task(input_dict)
        else:
            logger.info("Processing as CREATE task")
            result = process_create_task(input_dict)
    except Exception as e:
        logger.error(f"Error in agent_main, falling back to fix task: {e}")
        result = process_fix_task(input_dict)

    logger.info(f"agent_main completed, result length: {len(result) if result else 0}")
    os.system("git reset --hard")
    return result
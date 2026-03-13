from typing import List
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.agent_prompt import (
    AgentPrompt,
    AgentPromptCreate,
    AgentPromptUpdate,
    AgentPromptExecuteRequest,
    AgentPromptExecuteResponse
)
from app.services.agent_prompts_db import get_agent_prompts_db
from app.services.agent_executor import execute_agent_prompt, execute_agent_prompt_stream

router = APIRouter()


@router.get("/agent-prompts", response_model=List[AgentPrompt])
async def list_agent_prompts():
    """
    List all agent prompts (cross-notebook).

    Returns:
        List of agent prompts
    """
    try:
        db = get_agent_prompts_db()
        prompts = db.get_all_agent_prompts()

        return [AgentPrompt(**p) for p in prompts]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing agent prompts: {str(e)}"
        )


@router.post("/agent-prompts", response_model=AgentPrompt, status_code=status.HTTP_201_CREATED)
async def create_agent_prompt_endpoint(request: AgentPromptCreate):
    """
    Create a new agent prompt (cross-notebook).

    Args:
        request: Agent prompt creation data

    Returns:
        Created agent prompt
    """
    try:
        db = get_agent_prompts_db()

        # Convert Pydantic models to dicts for storage
        variables_data = [v.dict() for v in request.variables] if request.variables else []

        prompt_id = db.create_agent_prompt(
            name=request.name,
            description=request.description,
            icon=request.icon or "Bot",
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            template_prompt=request.template_prompt,
            variables=variables_data
        )

        prompt = db.get_agent_prompt(prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error retrieving created agent prompt"
            )

        return AgentPrompt(**prompt)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating agent prompt: {str(e)}"
        )


@router.get("/agent-prompts/{prompt_id}", response_model=AgentPrompt)
async def get_agent_prompt_endpoint(prompt_id: str):
    """
    Get an agent prompt by ID.

    Args:
        prompt_id: Agent prompt UUID

    Returns:
        Agent prompt
    """
    try:
        db = get_agent_prompts_db()
        prompt = db.get_agent_prompt(prompt_id)

        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent prompt not found"
            )

        return AgentPrompt(**prompt)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving agent prompt: {str(e)}"
        )


@router.put("/agent-prompts/{prompt_id}", response_model=AgentPrompt)
async def update_agent_prompt_endpoint(prompt_id: str, request: AgentPromptUpdate):
    """
    Update an agent prompt.

    Args:
        prompt_id: Agent prompt UUID
        request: Update data

    Returns:
        Updated agent prompt
    """
    try:
        db = get_agent_prompts_db()

        # Check if prompt exists
        existing = db.get_agent_prompt(prompt_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent prompt not found"
            )

        # Prepare update data
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.icon is not None:
            update_data["icon"] = request.icon
        if request.system_prompt is not None:
            update_data["system_prompt"] = request.system_prompt
        if request.user_prompt is not None:
            update_data["user_prompt"] = request.user_prompt
        if request.template_prompt is not None:
            update_data["template_prompt"] = request.template_prompt
        if request.variables is not None:
            update_data["variables"] = [v.dict() for v in request.variables]

        # Update prompt
        success = db.update_agent_prompt(prompt_id=prompt_id, **update_data)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error updating agent prompt"
            )

        # Get updated prompt
        prompt = db.get_agent_prompt(prompt_id)
        return AgentPrompt(**prompt)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating agent prompt: {str(e)}"
        )


@router.delete("/agent-prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_prompt_endpoint(prompt_id: str):
    """
    Delete an agent prompt.

    Args:
        prompt_id: Agent prompt UUID
    """
    try:
        db = get_agent_prompts_db()

        # Check if prompt exists
        existing = db.get_agent_prompt(prompt_id)
        if not existing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent prompt not found"
            )

        success = db.delete_agent_prompt(prompt_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Error deleting agent prompt"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting agent prompt: {str(e)}"
        )


@router.post("/agent-prompts/{prompt_id}/execute", response_model=AgentPromptExecuteResponse)
async def execute_agent_prompt_endpoint(prompt_id: str, request: AgentPromptExecuteRequest):
    """
    Execute an agent prompt on selected documents.

    Creates a new conversation with the result.

    Args:
        prompt_id: Agent prompt UUID
        request: Execution request with document_ids, notebook_id, and variable_values

    Returns:
        New conversation ID and title
    """
    try:
        db = get_agent_prompts_db()

        # Get the prompt
        prompt = db.get_agent_prompt(prompt_id)
        if not prompt:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent prompt not found"
            )

        if not request.document_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No documents selected"
            )

        # Execute the agent prompt
        result = await execute_agent_prompt(
            agent_prompt=prompt,
            document_ids=request.document_ids,
            notebook_id=request.notebook_id,
            variable_values=request.variable_values
        )

        return AgentPromptExecuteResponse(
            conversation_id=result["conversation_id"],
            title=result["title"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing agent prompt: {str(e)}"
        )


@router.post("/agent-prompts/{prompt_id}/execute_stream")
async def execute_agent_prompt_stream_endpoint(prompt_id: str, request: AgentPromptExecuteRequest):
    """
    Execute an agent prompt on selected documents with SSE streaming.

    Returns a streaming response with real-time token delivery.

    Args:
        prompt_id: Agent prompt UUID
        request: Execution request with document_ids, notebook_id, and variable_values

    Returns:
        StreamingResponse with SSE events
    """
    db = get_agent_prompts_db()

    prompt = db.get_agent_prompt(prompt_id)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent prompt not found"
        )

    if not request.document_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents selected"
        )

    return StreamingResponse(
        execute_agent_prompt_stream(
            agent_prompt=prompt,
            document_ids=request.document_ids,
            notebook_id=request.notebook_id,
            variable_values=request.variable_values
        ),
        media_type="text/event-stream"
    )

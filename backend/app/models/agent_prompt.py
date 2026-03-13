from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class PromptVariable(BaseModel):
    """Definition of a dynamic variable for the prompt."""
    key: str = Field(..., description="Variable key used in prompt templates, e.g., 'framework'")
    label: str = Field(..., description="Display label in UI")
    default: Optional[str] = None
    required: bool = False
    placeholder: Optional[str] = None


class AgentPromptCreate(BaseModel):
    """Model for creating a new agent prompt."""
    name: str
    description: Optional[str] = None
    icon: Optional[str] = "Bot"
    system_prompt: str
    user_prompt: str
    template_prompt: Optional[str] = None
    variables: List[PromptVariable] = []


class AgentPromptUpdate(BaseModel):
    """Model for updating an agent prompt."""
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    template_prompt: Optional[str] = None
    variables: Optional[List[PromptVariable]] = None


class AgentPrompt(BaseModel):
    """Full agent prompt model."""
    id: str
    name: str
    description: Optional[str]
    icon: str
    system_prompt: str
    user_prompt: str
    template_prompt: Optional[str]
    variables: List[PromptVariable]
    created_at: datetime
    updated_at: datetime


class AgentPromptExecuteRequest(BaseModel):
    """Request model for executing an agent prompt."""
    document_ids: List[str]
    notebook_id: str
    variable_values: Dict[str, Any] = {}  # Values for dynamic variables


class AgentPromptExecuteResponse(BaseModel):
    """Response model for agent prompt execution."""
    conversation_id: str
    title: str

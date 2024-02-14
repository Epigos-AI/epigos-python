import dataclasses

from epigos import typings


@dataclasses.dataclass
class Project:
    """Project data class"""

    id: str
    workspace_id: str
    name: str
    project_type: typings.ProjectType

from typing import Any

from pydantic import BaseModel


class FluxwayBaseSchema(BaseModel):
    def as_dict(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)

from typing import Literal

from pydantic import BaseModel, Field


class CreatePaymentRequest(BaseModel):
    currency: Literal["USD"] | None = Field(
        default=None,
        description="The ledger supports USD only; amounts are USD cents. May be omitted for legacy requests.",
    )
    merchant: str = Field(default="", max_length=200)
    merchant_category: str = Field(default="unknown", min_length=1, max_length=100)
    description: str = Field(default="", max_length=1000)

    user_id: str = Field(
        ...,
        min_length=5,
        max_length=50,
        json_schema_extra={"example": "user_123"},
    )

    amount: int = Field(
        ...,
        gt=0,
        json_schema_extra={"example": 100},
    )

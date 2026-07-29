"""Synthetic enterprise ticket domain with known-correct answers."""

from .tickets import TICKETS, Ticket
from .env import TicketEnv, build_registry

__all__ = ["TICKETS", "Ticket", "TicketEnv", "build_registry"]

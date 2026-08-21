"""AI Smart Scope Aiming Engine.

Consumes bridge-layer tracker output and produces an advisory aim solution and
HUD data for a human operator. Does not perform detection or tracking
(see project pipeline docs), and does not fire anything — this is an
aim-assist system, not an autonomous weapon.
"""

from .aiming_manager import AimingManager

__all__ = ["AimingManager"]

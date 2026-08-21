"""Bridge layer: adapts the OpticalFlowTracker 2D tracker plus a laser
rangefinder and gimbal/IMU pose source into aiming_engine.TrackerFrame
objects.
"""

from .frame_builder import TrackerFrameBuilder

__all__ = ["TrackerFrameBuilder"]

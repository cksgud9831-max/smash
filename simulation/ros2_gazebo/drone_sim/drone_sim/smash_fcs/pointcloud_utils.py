"""sensor_msgs/PointCloud2 원시 버퍼에서 각 점의 경사거리를 추출한다.

ROS 메시지 타입에 의존하지 않도록 원시 값(바이트 버퍼 + 필드 오프셋)만 받는다.
덕분에 ROS 없이도 단위 검증할 수 있다.

가제보 gpu_lidar 가 <topic>/points 로 내보내는 PointCloudPacked 는 브리지를 거쳐
x, y, z (float32) 필드를 가진 PointCloud2 가 된다. 반사가 없는 빔은 inf 또는 NaN
으로 들어오므로 걸러 낸다.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# sensor_msgs/PointField 데이터타입 코드 -> numpy 타입 문자
_POINTFIELD_DTYPES: Mapping[int, str] = {
    1: "i1",  # INT8
    2: "u1",  # UINT8
    3: "i2",  # INT16
    4: "u2",  # UINT16
    5: "i4",  # INT32
    6: "u4",  # UINT32
    7: "f4",  # FLOAT32
    8: "f8",  # FLOAT64
}


def ranges_from_xyz_buffer(
    data: bytes,
    point_step: int,
    x_offset: int,
    y_offset: int,
    z_offset: int,
    datatype: int = 7,
    is_bigendian: bool = False,
) -> np.ndarray:
    """PointCloud2 버퍼에서 각 점의 원점 기준 거리 배열을 만든다.

    유한하지 않거나 원점과 겹치는 점은 제외한다. 반환 배열이 비어 있으면
    유효 반사가 하나도 없었다는 뜻이다.
    """

    if point_step <= 0 or not data:
        return np.empty(0, dtype=np.float64)
    if datatype not in _POINTFIELD_DTYPES:
        raise ValueError(f"지원하지 않는 PointField datatype: {datatype}")

    dtype = np.dtype(("<" if not is_bigendian else ">") + _POINTFIELD_DTYPES[datatype])
    item_size = dtype.itemsize

    raw = np.frombuffer(data, dtype=np.uint8)
    n_points = raw.size // point_step
    if n_points == 0:
        return np.empty(0, dtype=np.float64)
    raw = raw[: n_points * point_step].reshape(n_points, point_step)

    def column(offset: int) -> np.ndarray:
        if offset + item_size > point_step:
            raise ValueError("필드 오프셋이 point_step 을 넘어섭니다")
        # np.frombuffer 는 읽기 전용 뷰이므로 view() 전에 복사가 필요하다.
        return np.ascontiguousarray(raw[:, offset : offset + item_size]).view(dtype).reshape(-1)

    x = column(x_offset).astype(np.float64)
    y = column(y_offset).astype(np.float64)
    z = column(z_offset).astype(np.float64)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if not np.any(finite):
        return np.empty(0, dtype=np.float64)

    distances = np.sqrt(x[finite] ** 2 + y[finite] ** 2 + z[finite] ** 2)
    return distances[distances > 0.0]


def field_offsets(fields) -> tuple[int, int, int, int]:
    """PointCloud2.fields 목록에서 (x_offset, y_offset, z_offset, datatype) 를 찾는다."""

    lookup = {f.name: f for f in fields}
    missing = [name for name in ("x", "y", "z") if name not in lookup]
    if missing:
        raise ValueError(f"PointCloud2 에 필드가 없습니다: {missing}")
    return (
        int(lookup["x"].offset),
        int(lookup["y"].offset),
        int(lookup["z"].offset),
        int(lookup["x"].datatype),
    )

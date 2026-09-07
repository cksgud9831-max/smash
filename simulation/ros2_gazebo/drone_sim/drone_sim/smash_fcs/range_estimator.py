"""표적까지의 거리(Range) 추정: 동축 광각 레이저 우선, 바운딩박스 기하 역추정 폴백.

■ 왜 이중화인가
    1. 가제보 LiDAR 는 물리적으로 정확하지만 시야가 좁다. 표적이 빔 원뿔 밖에 있는
       초기 포착 단계에서는 값을 주지 못하거나 배경을 읽는다.
    2. 바운딩박스 기하 역추정은 표적이 화면 안에 있기만 하면 항상 값을 주지만,
       표적의 실제 물리 크기 가정에 의존하는 계통오차를 가진다.

    => 초기 포착 단계는 바운딩박스 추정으로 리드각을 대략 만들어 포탑을 표적 쪽으로
       끌고 오고, 표적이 빔 원뿔 안에 들어오면 레이저로 전환해 정밀 거리를 얻는다.

■ 왜 단일 보어사이트 빔이 아니라 원뿔인가 (설계상 핵심)
    카메라 광학중심과 포구는 gun_link 기준 (0.572, -0.21, -0.292) m 떨어져 있고,
    보어사이트에 수직인 성분이 약 0.359 m 다. 포탑이 탄도해대로 지향하면 카메라
    시선과 포신 지향선 사이에 시차각 약 0.359/R [rad] 이 생긴다. R = 35 m 에서
    약 10 mrad 이며, 포탑 회전축까지 포함한 실제 시차는 오프라인 검증(A3)에서
    15.6 mrad(0.90도) 로 측정되었다. 반면 35 m 거리의 소형 드론은 화면에서
    13.4 mrad(약 21픽셀)에 불과하다.

    즉 포신이 정확히 조준된 순간 표적은 카메라 주점 밖에 있으므로, 주점에 고정된
    단일 빔은 표적을 놓치고 배경을 읽는다. 초기 구현에서 실제로 전 프레임 무효였다.

    해결책은 빔을 시차 + 리드각 + 추적오차를 덮는 원뿔(+-0.04 rad)로 넓히고
    원뿔 안의 최근접 유효 반사(first return)를 표적 거리로 쓰는 것이다. 실제
    펄스식 레이저 거리측정기의 first-return 논리와 같다. 하늘은 반사가 없고
    표적은 배경보다 항상 가까우므로 최근접 반사가 곧 표적이다.

    다만 원뿔이 넓어진 만큼 다른 물체(다른 드론, 조류, 저앙각의 지면)를 물 수
    있으므로, 바운딩박스 추정치와의 교차검증 게이트를 한 번 더 건다.

■ 바운딩박스 역추정 수식
    핀홀 모델에서 거리 R 에 있는 물리 크기 S 의 물체는 픽셀 크기 s = f * S / R 로
    맺힌다. 따라서

        R = f * S / s

    오차 전파: R = f*S/s 이므로 상대오차는 다음과 같이 결합된다.

        (dR/R)^2 = (dS/S)^2 + (ds/s)^2

    바운딩박스가 작아질수록(= 멀수록) 픽셀 1개의 오차가 거리 오차를 크게 키운다.
    이 sigma 를 함께 반환해 상위 로직이 신뢰도를 판단하고 교차검증 게이트를
    걸 수 있게 한다.

■ 알려진 한계 (투명성)
    1. S 는 표적 종류마다 다르고, 같은 표적도 시선 방향에 따라 투영 폭이 달라지므로
       계통오차가 존재한다. 본 추정기는 이를 보정하지 않는다.
    2. YOLO 바운딩박스는 광학흐름 추종 중 크기 스무딩(BBOX_SIZE_ALPHA)을 받으므로
       s 는 약간 지연된 값이다.
    3. 원뿔 레이저는 표적 표면 중 가장 가까운 점까지의 거리를 준다. 표적 중심까지의
       거리보다 표적 반경(수십 cm)만큼 짧다. 35 m 교전에서 이는 약 0.7% 이며
       탄착 시간 오차로는 마이크로초 수준이라 무시했다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

# 거리 출처 라벨
SOURCE_LASER = "laser"
SOURCE_BBOX = "bbox"
SOURCE_NONE = "none"


@dataclass(frozen=True)
class RangeEstimate:
    distance_m: float
    sigma_m: float
    source: str
    valid: bool

    @staticmethod
    def invalid(reason: str = SOURCE_NONE) -> "RangeEstimate":
        return RangeEstimate(distance_m=0.0, sigma_m=float("inf"), source=reason, valid=False)


@dataclass(frozen=True)
class LaserConeSample:
    """광각 레이저 한 스캔. 각 빔의 경사거리[m] 목록만 담는다.

    ROS PointCloud2 를 파싱해 각 점의 노름을 구한 결과를 넣으면 된다.
    반사가 없는 빔(하늘)은 inf 또는 NaN 으로 들어오며 read() 에서 걸러진다.
    """

    ranges: tuple[float, ...]
    timestamp: float


def bbox_contains(bbox_xyxy: tuple[float, float, float, float], point: tuple[float, float]) -> bool:
    x1, y1, x2, y2 = bbox_xyxy
    u, v = point
    return (x1 <= u <= x2) and (y1 <= v <= y2)


class ConeLaserRange:
    """동축 광각 레이저. 최신 스캔의 최근접 유효 반사를 표적 거리로 본다."""

    def __init__(
        self,
        staleness_timeout_s: float = 0.2,
        noise_sigma_m: float = 0.02,
        range_min_m: float = 1.0,
        range_max_m: float = 500.0,
    ):
        self._staleness_timeout_s = float(staleness_timeout_s)
        self._noise_sigma_m = float(noise_sigma_m)
        self._range_min_m = float(range_min_m)
        self._range_max_m = float(range_max_m)
        self._latest: Optional[LaserConeSample] = None

    def submit(self, sample: LaserConeSample) -> None:
        self._latest = sample

    def submit_ranges(self, ranges: Sequence[float], timestamp: float) -> None:
        self.submit(LaserConeSample(ranges=tuple(float(r) for r in ranges), timestamp=float(timestamp)))

    def read(self, now: float) -> RangeEstimate:
        sample = self._latest
        if sample is None:
            return RangeEstimate.invalid()
        if now - sample.timestamp > self._staleness_timeout_s:
            return RangeEstimate.invalid()

        nearest = math.inf
        for r in sample.ranges:
            if math.isfinite(r) and self._range_min_m <= r <= self._range_max_m and r < nearest:
                nearest = r
        if not math.isfinite(nearest):
            return RangeEstimate.invalid()

        return RangeEstimate(
            distance_m=float(nearest),
            sigma_m=self._noise_sigma_m,
            source=SOURCE_LASER,
            valid=True,
        )


class BoundingBoxRange:
    """단안 바운딩박스 기하 역추정."""

    def __init__(
        self,
        characteristic_size_m: float,
        size_uncertainty_ratio: float = 0.25,
        bbox_pixel_sigma: float = 2.0,
        min_bbox_px: float = 4.0,
        min_range_m: float = 1.0,
        max_range_m: float = 800.0,
    ):
        if characteristic_size_m <= 0.0:
            raise ValueError("characteristic_size_m 은 0보다 커야 합니다")
        self._size_m = float(characteristic_size_m)
        self._size_uncertainty_ratio = float(size_uncertainty_ratio)
        self._bbox_pixel_sigma = float(bbox_pixel_sigma)
        self._min_bbox_px = float(min_bbox_px)
        self._min_range_m = float(min_range_m)
        self._max_range_m = float(max_range_m)

    def estimate(
        self,
        bbox_xyxy: tuple[float, float, float, float],
        fx: float,
        fy: float,
    ) -> RangeEstimate:
        x1, y1, x2, y2 = bbox_xyxy
        w_px = abs(x2 - x1)
        h_px = abs(y2 - y1)
        size_px = max(w_px, h_px)
        if size_px < self._min_bbox_px:
            return RangeEstimate.invalid()

        # 가로/세로 중 큰 쪽을 썼으므로 대응하는 축의 초점거리를 쓴다.
        focal_px = fx if w_px >= h_px else fy
        if focal_px <= 0.0:
            return RangeEstimate.invalid()

        distance = focal_px * self._size_m / size_px
        if not (self._min_range_m <= distance <= self._max_range_m):
            return RangeEstimate.invalid()

        relative = math.hypot(self._size_uncertainty_ratio, self._bbox_pixel_sigma / size_px)
        return RangeEstimate(
            distance_m=float(distance),
            sigma_m=float(distance * relative),
            source=SOURCE_BBOX,
            valid=True,
        )


class HybridRangeEstimator:
    """레이저 우선 + 바운딩박스 교차검증/폴백.

    판정 순서
        1. 레이저와 바운딩박스가 둘 다 유효하고 두 값이 교차검증 게이트 안이면 레이저 채택
           (레이저가 훨씬 정밀하다: 2 cm 대 25%)
        2. 레이저만 유효하면(표적이 너무 작아 bbox 추정 불가 등) 레이저 채택
        3. 레이저가 게이트를 벗어나거나 무효면 바운딩박스 채택
        4. 둘 다 무효면 무효

    게이트를 벗어난 레이저 값은 표적이 아닌 다른 물체(다른 비행체, 조류, 저앙각의
    지면)를 물었다는 뜻이므로 버리는 것이 옳다.
    """

    def __init__(
        self,
        laser: ConeLaserRange,
        bbox: BoundingBoxRange,
        smoothing_alpha: float = 1.0,
        outlier_sigma_gate: float = 5.0,
        cross_check_sigma: float = 3.0,
        max_range_rate_mps: float = 60.0,
    ):
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha 는 (0, 1] 범위여야 합니다")
        self._laser = laser
        self._bbox = bbox
        self._alpha = float(smoothing_alpha)
        self._outlier_sigma_gate = float(outlier_sigma_gate)
        self._cross_check_sigma = float(cross_check_sigma)
        self._max_range_rate_mps = float(max_range_rate_mps)
        self._last: Optional[RangeEstimate] = None
        self._last_time: Optional[float] = None
        self._laser_rejections = 0
        self._jump_rejections = 0

    @property
    def laser(self) -> ConeLaserRange:
        return self._laser

    @property
    def laser_rejections(self) -> int:
        """교차검증 게이트에서 기각된 레이저 표본 수. 진단용."""

        return self._laser_rejections

    @property
    def jump_rejections(self) -> int:
        """프레임 간 급변 게이트에서 기각된 표본 수. 진단용."""

        return self._jump_rejections

    def reset(self) -> None:
        self._last = None
        self._last_time = None

    def estimate(
        self,
        now: float,
        bbox_xyxy: tuple[float, float, float, float],
        principal_point: tuple[float, float],
        fx: float,
        fy: float,
    ) -> RangeEstimate:
        # principal_point 는 현재 판정에 쓰이지 않지만, 향후 빔 방향별 선택 방식으로
        # 확장할 때 필요하므로 시그니처에 유지한다.
        del principal_point

        bbox_estimate = self._bbox.estimate(bbox_xyxy, fx, fy)
        laser_estimate = self._laser.read(now)

        result = RangeEstimate.invalid()
        if laser_estimate.valid and bbox_estimate.valid:
            gate = self._cross_check_sigma * max(bbox_estimate.sigma_m, 1e-6)
            if abs(laser_estimate.distance_m - bbox_estimate.distance_m) <= gate:
                result = laser_estimate
            else:
                self._laser_rejections += 1
                result = bbox_estimate
        elif laser_estimate.valid:
            result = laser_estimate
        elif bbox_estimate.valid:
            result = bbox_estimate

        if not result.valid:
            self._last = None
            return result

        previous = self._last
        if previous is not None and previous.source == result.source:
            # 프레임 간 급변 게이트.
            #
            # 게이트 폭은 측정 잡음만으로 정하면 안 된다. 표적이 실제로 접근/이탈
            # 중이면 거리는 프레임마다 정상적으로 변하기 때문이다. 레이저 잡음
            # sigma 는 2 cm 인데 20 m/s 표적은 30 fps 에서 프레임당 0.67 m 씩
            # 변하므로, 잡음 기준 게이트만 쓰면 정상적인 변화를 전부 이상치로
            # 기각해 거리값이 얼어붙는다(실제로 검증 중 이 현상을 확인했다).
            # 따라서 측정 잡음 항에 "물리적으로 가능한 최대 거리변화율 x 경과시간"
            # 항을 더한다.
            dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
            gate = (
                self._outlier_sigma_gate * max(previous.sigma_m, 1e-6)
                + self._max_range_rate_mps * dt
            )
            if abs(result.distance_m - previous.distance_m) > gate:
                # 같은 출처인데 물리적으로 불가능하게 튀었다면 이번 프레임만 버린다.
                self._jump_rejections += 1
                self._last = previous
                self._last_time = now
                return previous
            if self._alpha < 1.0:
                blended = self._alpha * result.distance_m + (1.0 - self._alpha) * previous.distance_m
                result = RangeEstimate(
                    distance_m=float(blended),
                    sigma_m=result.sigma_m,
                    source=result.source,
                    valid=True,
                )

        self._last = result
        self._last_time = now
        return result

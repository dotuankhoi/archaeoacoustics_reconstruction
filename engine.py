import numpy as np
from dataclasses import dataclass, field
from typing import Optional


SPEED_OF_SOUND = 343.0
AIR_ABSORPTION_PER_METER = 0.002


@dataclass
class Material:
    name: str
    absorption: float

    def reflection_coeff(self) -> float:
        return 1.0 - self.absorption


MATERIALS = {
    "stone":       Material("Bare Stone",        0.02),
    "brick":       Material("Brick",             0.03),
    "marble":      Material("Marble",            0.01),
    "plaster":     Material("Plaster",           0.05),
    "wood_panel":  Material("Wood Paneling",     0.10),
    "wood_floor":  Material("Wood Floor",        0.08),
    "tapestry":    Material("Heavy Tapestry",    0.55),
    "curtain":     Material("Thick Curtain",     0.45),
    "audience":    Material("Seated Audience",   0.70),
    "soil":        Material("Compacted Soil",    0.15),
    "concrete":    Material("Concrete",          0.02),
    "air_gap":     Material("Open Air",          1.00),
}


@dataclass
class Wall:
    x0: float
    y0: float
    x1: float
    y1: float
    material: Material

    def __post_init__(self):
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        length = np.hypot(dx, dy)
        self._normal = np.array([-dy / length, dx / length])
        self._vec = np.array([dx, dy])

    @property
    def normal(self) -> np.ndarray:
        return self._normal

    def intersect(self, origin: np.ndarray, direction: np.ndarray) -> Optional[float]:
        ox, oy = origin
        dx, dy = direction
        wx, wy = self.x0, self.y0
        vx, vy = self._vec

        denom = dx * vy - dy * vx
        if abs(denom) < 1e-12:
            return None

        t = ((wx - ox) * vy - (wy - oy) * vx) / denom
        s = ((wx - ox) * dy - (wy - oy) * dx) / denom

        if t > 1e-9 and 0.0 <= s <= 1.0:
            return t
        return None


@dataclass
class Room:
    walls: list[Wall] = field(default_factory=list)

    def add_wall(self, x0, y0, x1, y1, material_key: str = "stone"):
        self.walls.append(Wall(x0, y0, x1, y1, MATERIALS[material_key]))

    def add_rect(self, x0, y0, x1, y1,
                 floor_mat="wood_floor", ceiling_mat="plaster",
                 left_mat="stone", right_mat="stone"):
        self.add_wall(x0, y0, x1, y0, floor_mat)
        self.add_wall(x1, y0, x1, y1, right_mat)
        self.add_wall(x1, y1, x0, y1, ceiling_mat)
        self.add_wall(x0, y1, x0, y0, left_mat)


@dataclass
class RayHit:
    time: float
    energy: float
    pan: float = 0.0


class AcousticRayTracer:
    def __init__(self, room: Room, source: tuple[float, float],
                 receiver: tuple[float, float],
                 n_rays: int = 2000,
                 max_bounces: int = 50,
                 max_time: float = 2.0,
                 receiver_radius: float = 0.25,
                 min_energy: float = 1e-5):
        self.room = room
        self.source = np.array(source, dtype=float)
        self.receiver = np.array(receiver, dtype=float)
        self.n_rays = n_rays
        self.max_bounces = max_bounces
        self.max_time = max_time
        self.receiver_radius = receiver_radius
        self.min_energy = min_energy

    def _reflect(self, direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
        return direction - 2.0 * np.dot(direction, normal) * normal

    def _check_receiver_crossing(self, origin, direction, t_wall) -> Optional[float]:
        oc = self.receiver - origin
        t_proj = np.dot(oc, direction)
        t_proj = np.clip(t_proj, 0.0, t_wall)
        closest = origin + t_proj * direction
        dist = np.linalg.norm(closest - self.receiver)
        if dist <= self.receiver_radius:
            return np.linalg.norm(t_proj * direction)
        return None

    def _pan(self, direction: np.ndarray) -> float:
        arrival = -direction
        return float(self._facing[0] * arrival[1] - self._facing[1] * arrival[0])

    def trace(self) -> list[RayHit]:
        hits: list[RayHit] = []
        angles = np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)

        to_source = self.source - self.receiver
        norm = np.linalg.norm(to_source)
        self._facing = to_source / norm if norm > 0 else np.array([1.0, 0.0])

        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            origin = self.source.copy()
            energy = 1.0
            path_length = 0.0

            oc = self.receiver - origin
            t_direct = np.dot(oc, direction)
            if t_direct > 0:
                closest = origin + t_direct * direction
                if np.linalg.norm(closest - self.receiver) <= self.receiver_radius:
                    direct_len = np.linalg.norm(self.receiver - origin)
                    t_arrive = direct_len / SPEED_OF_SOUND
                    air_loss = np.exp(-AIR_ABSORPTION_PER_METER * direct_len)
                    spread = 1.0 / max(direct_len ** 1.5, 0.01)
                    if t_arrive <= self.max_time:
                        hits.append(RayHit(t_arrive, energy * air_loss * spread,
                                           self._pan(direction)))

            for _ in range(self.max_bounces):
                if energy < self.min_energy:
                    break

                t_min = np.inf
                hit_wall: Optional[Wall] = None
                for wall in self.room.walls:
                    t = wall.intersect(origin, direction)
                    if t is not None and t < t_min:
                        t_min = t
                        hit_wall = wall

                if hit_wall is None or t_min == np.inf:
                    break

                recv_t = self._check_receiver_crossing(origin, direction, t_min)
                if recv_t is not None:
                    seg_len = recv_t
                    total_len = path_length + seg_len
                    t_arrive = total_len / SPEED_OF_SOUND
                    air_loss = np.exp(-AIR_ABSORPTION_PER_METER * total_len)
                    spread = 1.0 / max(total_len ** 1.5, 0.01)
                    if t_arrive <= self.max_time:
                        hits.append(RayHit(t_arrive, energy * air_loss * spread,
                                           self._pan(direction)))

                hit_point = origin + t_min * direction
                path_length += t_min

                if path_length / SPEED_OF_SOUND > self.max_time:
                    break

                energy *= hit_wall.material.reflection_coeff()

                normal = hit_wall.normal
                if np.dot(direction, normal) > 0:
                    normal = -normal
                direction = self._reflect(direction, normal)
                direction /= np.linalg.norm(direction)

                origin = hit_point + direction * 1e-6

        return hits


MAX_ITD_SECONDS = 0.00066


def build_impulse_response(hits: list[RayHit],
                            sample_rate: int = 44100,
                            max_time: float = 2.0) -> np.ndarray:
    n_samples = int(max_time * sample_rate)
    ir = np.zeros((n_samples, 2), dtype=np.float64)

    for hit in hits:
        sample_idx = int(hit.time * sample_rate)
        if not (0 <= sample_idx < n_samples):
            continue

        pan = float(np.clip(hit.pan, -1.0, 1.0))
        gain_l = np.cos((pan + 1.0) * np.pi / 4.0)
        gain_r = np.sin((pan + 1.0) * np.pi / 4.0)

        itd = int(round(abs(pan) * MAX_ITD_SECONDS * sample_rate))
        idx_l = sample_idx + (itd if pan > 0 else 0)
        idx_r = sample_idx + (itd if pan < 0 else 0)

        if idx_l < n_samples:
            ir[idx_l, 0] += hit.energy * gain_l
        if idx_r < n_samples:
            ir[idx_r, 1] += hit.energy * gain_r

    if len(hits) > 0:
        window = np.hanning(7)
        window /= window.sum()
        from scipy.signal import fftconvolve
        ir[:, 0] = fftconvolve(ir[:, 0], window, mode="same")
        ir[:, 1] = fftconvolve(ir[:, 1], window, mode="same")

    peak = np.max(np.abs(ir))
    if peak > 0:
        ir /= peak

    return ir

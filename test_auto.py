from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CarState:
    timestamp_us: int
    car_id: str
    x: float
    y: float
    theta: float
    dx: float
    dy: float
    angular_velocity: float
    u: int
    v: int

    def __post_init__(self):
        if not self.car_id or any(
            char in self.car_id for char in '\r\n":,'
        ):
            raise ValueError(
                "Die Fahrzeug-ID darf keine Anführungszeichen, "
                "Doppelpunkte, Kommas oder Zeilenumbrüche enthalten."
            )

    def to_bytes(self) -> bytes:
        """Erzeugt einen UTF-8-Datensatz im Aufgabenformat."""
        message = (
            f'{self.timestamp_us}:"{self.car_id}",'
            f"{self.x:.3f},{self.y:.3f},{self.theta:.3f},"
            f"{self.dx:.3f},{self.dy:.3f},"
            f"{self.angular_velocity:.3f},"
            f"{self.u},{self.v}\n"
        )

        payload = message.encode("utf-8")

        if len(payload) > 65507:
            raise ValueError("Datensatz ist zu groß für ein UDP-Paket.")

        return payload

    @classmethod
    def missing(cls, timestamp_us: int, car_id: str):
        """Fehlendes Fahrzeug; unbekannte Zusatzwerte sind nan bzw. -1."""
        return cls(
            timestamp_us=timestamp_us,
            car_id=car_id,
            x=-1000.0,
            y=-1000.0,
            theta=math.nan,
            dx=math.nan,
            dy=math.nan,
            angular_velocity=math.nan,
            u=-1,
            v=-1,
        )
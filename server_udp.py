import argparse
import logging
import math
import socket
import time

from test_auto import CarState


logger = logging.getLogger(__name__)


class UdpPublisher:
    """Sendet Fahrzeugdaten an einen Controller."""

    def __init__(self, host: str, port: int):
        # Adresse einmal auflösen, nicht bei jedem Frame.
        self.destination = socket.getaddrinfo(
            host,
            port,
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )[0][4]

        self.socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        )

    def publish(self, state: CarState):
        self.socket.sendto(
            state.to_bytes(),
            self.destination,
        )

    def close(self):
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def valid_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Der Port muss eine ganze Zahl sein."
        )

    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError(
            "Der Port muss zwischen 1 und 65535 liegen."
        )

    return port


def valid_fps(value: str) -> float:
    try:
        fps = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "FPS muss eine Zahl sein."
        )

    if not math.isfinite(fps) or not 0 < fps <= 1000:
        raise argparse.ArgumentTypeError(
            "FPS muss größer als 0 und höchstens 1000 sein."
        )

    return fps


def valid_frame_count(value: str) -> int:
    try:
        frames = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Die Frame-Anzahl muss eine ganze Zahl sein."
        )

    if frames < 0:
        raise argparse.ArgumentTypeError(
            "Die Frame-Anzahl darf nicht negativ sein."
        )

    return frames


def demo_states(fps: float, frames: int):
    """
    Erzeugt synthetische Fahrzeugdaten zum Testen des UDP-Versands.

    frames=0 bedeutet: unbegrenzt laufen.
    Die Pixelkoordinaten sind ebenfalls synthetisch.
    """
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns
    interval_ns = int(1_000_000_000 / fps)
    frame_index = 0

    while frames == 0 or frame_index < frames:
        now_ns = time.perf_counter_ns()

        if now_ns < deadline_ns:
            time.sleep(
                (deadline_ns - now_ns) / 1_000_000_000
            )

        timestamp_us = (
            time.perf_counter_ns() - start_ns
        ) // 1000

        seconds = timestamp_us / 1_000_000

        # Synthetische Kreisbewegung.
        x = 1250 + 300 * math.cos(seconds)
        y = 750 + 300 * math.sin(seconds)

        yield CarState(
            timestamp_us=timestamp_us,
            car_id="Demo Racer",
            x=x,
            y=y,
            theta=math.degrees(
                seconds + math.pi / 2
            ) % 360,
            dx=-300 * math.sin(seconds),
            dy=300 * math.cos(seconds),
            angular_velocity=math.degrees(1),
            u=round(x / 2),
            v=round(y / 2),
        )

        frame_index += 1
        deadline_ns += interval_ns

        # Nach Verzögerungen keine Paketflut zum Aufholen senden.
        now_ns = time.perf_counter_ns()

        if deadline_ns < now_ns:
            deadline_ns = now_ns + interval_ns


def main():
    parser = argparse.ArgumentParser(
        description="Toy Race Car UDP Vision Server"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="IP-Adresse des Empfängers",
    )

    parser.add_argument(
        "--port",
        type=valid_port,
        default=5000,
        help="UDP-Zielport, standardmäßig 5000",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        required=True,
        help="Synthetische Testdaten senden",
    )

    parser.add_argument(
        "--fps",
        type=valid_fps,
        default=60.0,
        help="Sendehäufigkeit der Demo-Daten",
    )

    parser.add_argument(
        "--frames",
        type=valid_frame_count,
        default=0,
        help="Anzahl der Datensätze; 0 = unbegrenzt",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    logger.warning(
        "DEMO-MODUS: synthetische Daten, "
        "noch kein SAM-Tracking."
    )

    try:
        with UdpPublisher(args.host, args.port) as publisher:
            logger.info(
                "Sende an %s:%s — Beenden mit Ctrl+C",
                *publisher.destination,
            )

            for state in demo_states(args.fps, args.frames):
                publisher.publish(state)

    except KeyboardInterrupt:
        logger.info("Server beendet.")

    except OSError as error:
        logger.error("Netzwerkfehler: %s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
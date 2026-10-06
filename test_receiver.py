import argparse
import socket

from server_udp import valid_port


def main():
    parser = argparse.ArgumentParser(
        description="UDP-Testempfänger für Fahrzeugdaten"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Lokale IP; 0.0.0.0 für alle IPv4-Schnittstellen",
    )

    parser.add_argument(
        "--port",
        type=valid_port,
        default=5000,
        help="UDP-Empfangsport, standardmäßig 5000",
    )

    args = parser.parse_args()

    try:
        with socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM,
        ) as receiver:
            receiver.bind((args.host, args.port))

            print(
                f"Empfänger läuft auf {args.host}:{args.port}",
                flush=True,
            )

            while True:
                payload, sender = receiver.recvfrom(65535)

                try:
                    message = payload.decode("utf-8")
                except UnicodeDecodeError:
                    print(
                        f"Ungültiges UTF-8-Paket von {sender}"
                    )
                    continue

                print(
                    message,
                    end="" if message.endswith("\n") else "\n",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nEmpfänger beendet.")

    except OSError as error:
        print(f"Netzwerkfehler: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
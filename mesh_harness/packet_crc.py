
"""CRC32 framing and recovery for Reed-Solomon mesh packets.

Wire format:
    4 bytes: unsigned packet index, big-endian
    N bytes: Reed-Solomon encoded packet payload
    4 bytes: CRC32 covering index + payload
"""

import struct
import zlib

from .run_mesh_demo import decode_packets


INDEX_SIZE = 4
CRC_SIZE = 4


def frame_packet(index: int, payload: bytes) -> bytes:
    """Attach an index and CRC32 to a packet payload."""

    if type(index) is not int or not 0 <= index <= 0xFFFFFFFF:
        raise ValueError("Invalid packet index.")

    if not isinstance(payload, bytes):
        raise TypeError("Payload must be bytes.")

    header = struct.pack(">I", index)
    body = header + payload

    checksum = zlib.crc32(body)

    return body + struct.pack(">I", checksum)


def verify_packet(
    frame: bytes,
    expected_index: int,
    packet_size: int,
) -> bytes | None:
    """Return payload if valid, otherwise None."""

    if not isinstance(frame, bytes):
        return None

    expected_size = INDEX_SIZE + packet_size + CRC_SIZE

    if len(frame) != expected_size:
        return None

    body = frame[:-CRC_SIZE]
    stored_crc = struct.unpack(">I", frame[-CRC_SIZE:])[0]
    calculated_crc = zlib.crc32(body)

    if stored_crc != calculated_crc:
        return None

    received_index = struct.unpack(">I", body[:INDEX_SIZE])[0]

    if received_index != expected_index:
        return None

    return body[INDEX_SIZE:]


def recover_framed_packets(
    frames: list[bytes | None],
    n: int,
    k: int,
    original_length: int,
    packet_size: int = 200,
) -> bytes:
    """Recover using verified packets and known erasure slots."""

    packets = []
    present = []

    for index, frame in enumerate(frames):
        payload = (
            None
            if frame is None
            else verify_packet(frame, index, packet_size)
        )

        if payload is None:
            packets.append(bytes(packet_size))
            present.append(False)
        else:
            packets.append(payload)
            present.append(True)

    return decode_packets(
        packets,
        present,
        n=n,
        k=k,
        original_length=original_length,
        packet_size=packet_size,
    )

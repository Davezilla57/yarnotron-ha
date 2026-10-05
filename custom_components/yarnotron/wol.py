"""Wake-on-LAN: a magic packet to wake a sleeping Mac."""

from __future__ import annotations

import asyncio
import re

_MAC = re.compile(r"^[0-9a-f]{12}$")


def parse_mac(value: str) -> bytes:
    """Accept aa:bb:cc:dd:ee:ff, aa-bb-..., or aabbccddeeff. Raise ValueError otherwise."""
    cleaned = re.sub(r"[\s:.\-]", "", value.strip().casefold())
    if not _MAC.match(cleaned):
        raise ValueError(f"not a MAC address: {value!r}")
    return bytes.fromhex(cleaned)


def magic_packet(mac: bytes) -> bytes:
    return b"\xff" * 6 + mac * 16


async def send_magic_packet(mac: str, broadcast: str = "255.255.255.255", port: int = 9) -> None:
    packet = magic_packet(parse_mac(mac))
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(
        asyncio.DatagramProtocol, remote_addr=(broadcast, port), allow_broadcast=True
    )
    try:
        transport.sendto(packet)
    finally:
        transport.close()

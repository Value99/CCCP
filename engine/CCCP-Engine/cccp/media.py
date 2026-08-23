"""Safe media loading and optional image/video decoding."""

from __future__ import annotations

import base64
import binascii
import http.client
import io
import ipaddress
import socket
import ssl
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable
from urllib.parse import unquote_to_bytes, urljoin, urlsplit


DEFAULT_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_REDIRECTS = 3


class MediaError(ValueError):
    """Base error for rejected or invalid media."""


class MediaSecurityError(MediaError):
    """The media source violates the network security policy."""


class MediaTooLargeError(MediaError):
    """The media exceeds the configured byte limit."""


class MediaDecodeError(MediaError):
    """The media payload cannot be decoded."""


@dataclass(frozen=True)
class MediaPayload:
    data: bytes
    mime_type: str
    source: str


@dataclass(frozen=True)
class VideoFrame:
    image: object
    timestamp: float


def _normalized_mime(value: str | None) -> str:
    return (value or "").split(";", 1)[0].strip().lower()


def _check_mime(mime_type: str, allowed_mime_prefixes: Iterable[str]) -> str:
    normalized = _normalized_mime(mime_type)
    prefixes = tuple(prefix.lower() for prefix in allowed_mime_prefixes)
    if not normalized or not any(normalized.startswith(prefix) for prefix in prefixes):
        raise MediaSecurityError(f"unsupported media MIME type: {mime_type!r}")
    return normalized


def _decode_data_uri(
    uri: str,
    *,
    max_bytes: int,
    allowed_mime_prefixes: Iterable[str],
) -> MediaPayload:
    header, separator, encoded = uri.partition(",")
    if not separator:
        raise MediaError("malformed data URI")
    metadata = header[5:].split(";")
    mime_type = _check_mime(metadata[0], allowed_mime_prefixes)
    parameters = {part.lower() for part in metadata[1:] if part}
    try:
        if "base64" in parameters:
            estimated_size = (len(encoded) * 3) // 4
            if estimated_size > max_bytes + 2:
                raise MediaTooLargeError(
                    f"media exceeds {max_bytes} bytes"
                )
            data = base64.b64decode(encoded, validate=True)
        else:
            data = unquote_to_bytes(encoded)
    except MediaTooLargeError:
        raise
    except (ValueError, binascii.Error) as exc:
        raise MediaError("invalid data URI payload") from exc
    if len(data) > max_bytes:
        raise MediaTooLargeError(f"media exceeds {max_bytes} bytes")
    return MediaPayload(data=data, mime_type=mime_type, source="data URI")


def _is_public_address(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return bool(ip.is_global)


def _resolve_public(hostname: str, port: int) -> list[str]:
    try:
        records = socket.getaddrinfo(
            hostname,
            port,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise MediaSecurityError(f"cannot resolve media host: {hostname}") from exc
    addresses = list(dict.fromkeys(record[4][0] for record in records))
    if not addresses or any(not _is_public_address(value) for value in addresses):
        raise MediaSecurityError(
            f"media host resolves to a non-public address: {hostname}"
        )
    return addresses


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        hostname: str,
        address: str,
        port: int,
        timeout: float,
    ) -> None:
        super().__init__(hostname, port=port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._address, self.port),
            self.timeout,
            self.source_address,
        )
        if self._tunnel_host:
            self.sock = raw_socket
            self._tunnel()
        self.sock = self._context.wrap_socket(
            raw_socket,
            server_hostname=self.host,
        )


def _request_once(
    url: str,
    *,
    timeout: float,
    max_bytes: int,
    allowed_mime_prefixes: Iterable[str],
) -> tuple[MediaPayload | None, str | None]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise MediaSecurityError("only data, HTTP, and HTTPS media URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise MediaSecurityError("media URL must contain a host and no credentials")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise MediaSecurityError("invalid media URL port") from exc
    addresses = _resolve_public(parsed.hostname, port)
    address = addresses[0]
    host_header = parsed.hostname
    default_port = 443 if parsed.scheme == "https" else 80
    if port != default_port:
        host_header = f"{host_header}:{port}"
    target = parsed.path or "/"
    if parsed.query:
        target += f"?{parsed.query}"
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = _PinnedHTTPSConnection(
            parsed.hostname,
            address,
            port,
            timeout,
        )
    else:
        connection = http.client.HTTPConnection(address, port, timeout=timeout)
    try:
        connection.request(
            "GET",
            target,
            headers={
                "Host": host_header,
                "Accept": "image/*, video/*",
                "User-Agent": "cccp-media/1",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        if response.status in {301, 302, 303, 307, 308}:
            location = response.getheader("Location")
            if not location:
                raise MediaError("media redirect is missing Location")
            return None, urljoin(url, location)
        if response.status < 200 or response.status >= 300:
            raise MediaError(f"media server returned HTTP {response.status}")
        mime_type = _check_mime(
            response.getheader("Content-Type"),
            allowed_mime_prefixes,
        )
        content_length = response.getheader("Content-Length")
        if content_length:
            try:
                if int(content_length) > max_bytes:
                    raise MediaTooLargeError(
                        f"media exceeds {max_bytes} bytes"
                    )
            except ValueError as exc:
                raise MediaError("invalid media Content-Length") from exc
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = response.read(min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise MediaTooLargeError(f"media exceeds {max_bytes} bytes")
        return MediaPayload(data=data, mime_type=mime_type, source=url), None
    except (OSError, http.client.HTTPException, socket.timeout) as exc:
        raise MediaError(f"failed to fetch media: {exc}") from exc
    finally:
        connection.close()


def load_media(
    source: str,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: float = DEFAULT_TIMEOUT,
    allowed_mime_prefixes: Iterable[str] = ("image/", "video/"),
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
) -> MediaPayload:
    """Load a data URI or public HTTP(S) URL under strict resource limits."""
    if not isinstance(source, str):
        raise TypeError("media source must be a string")
    if max_bytes <= 0 or timeout <= 0 or max_redirects < 0:
        raise ValueError("media limits must be positive")
    if source.startswith("data:"):
        return _decode_data_uri(
            source,
            max_bytes=max_bytes,
            allowed_mime_prefixes=allowed_mime_prefixes,
        )
    current = source
    for redirect_count in range(max_redirects + 1):
        payload, redirect = _request_once(
            current,
            timeout=timeout,
            max_bytes=max_bytes,
            allowed_mime_prefixes=allowed_mime_prefixes,
        )
        if payload is not None:
            return payload
        if redirect_count == max_redirects:
            raise MediaSecurityError("too many media redirects")
        current = redirect or ""
    raise AssertionError("unreachable")


def decode_image(payload: MediaPayload, *, max_pixels: int = 64_000_000):
    """Decode an image payload with Pillow and return an owned RGB image."""
    if not payload.mime_type.startswith("image/"):
        raise MediaDecodeError("payload is not an image")
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:
        raise MediaDecodeError("Pillow is required for image decoding") from exc
    try:
        with Image.open(io.BytesIO(payload.data)) as image:
            if image.width <= 0 or image.height <= 0:
                raise MediaDecodeError("image has invalid dimensions")
            if image.width * image.height > max_pixels:
                raise MediaTooLargeError(
                    f"decoded image exceeds {max_pixels} pixels"
                )
            image.load()
            return image.convert("RGB")
    except MediaError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise MediaDecodeError("invalid image payload") from exc


def decode_video(
    payload: MediaPayload,
    *,
    sample_fps: float = 2.0,
    max_frames: int | None = None,
    max_pixels_per_frame: int = 16_000_000,
) -> list[VideoFrame]:
    """Decode and uniformly sample a video using optional PyAV."""
    if not payload.mime_type.startswith("video/"):
        raise MediaDecodeError("payload is not a video")
    if sample_fps <= 0 or (max_frames is not None and max_frames <= 0):
        raise ValueError("video sampling limits must be positive")
    try:
        import av
        from PIL import Image
    except ImportError as exc:
        raise MediaDecodeError("PyAV and Pillow are required for video decoding") from exc
    frames: list[VideoFrame] = []
    next_timestamp = 0.0
    try:
        with av.open(io.BytesIO(payload.data), mode="r") as container:
            streams = [stream for stream in container.streams if stream.type == "video"]
            if not streams:
                raise MediaDecodeError("video contains no video stream")
            stream = streams[0]
            for frame in container.decode(stream):
                timestamp = (
                    float(frame.time)
                    if frame.time is not None
                    else float(len(frames)) / float(stream.average_rate or sample_fps)
                )
                if timestamp + 1e-9 < next_timestamp:
                    continue
                if frame.width * frame.height > max_pixels_per_frame:
                    raise MediaTooLargeError(
                        f"decoded video frame exceeds {max_pixels_per_frame} pixels"
                    )
                array = frame.to_ndarray(format="rgb24")
                frames.append(VideoFrame(Image.fromarray(array, "RGB"), timestamp))
                next_timestamp = timestamp + 1.0 / sample_fps
                if max_frames is not None and len(frames) >= max_frames:
                    break
    except MediaError:
        raise
    except Exception as exc:
        raise MediaDecodeError("invalid video payload") from exc
    if not frames:
        raise MediaDecodeError("video contains no decodable frames")
    return frames


def canonical_media_reference(kind: str, source: str) -> dict[str, str]:
    """Return the stable, prompt-free identity of one ordered media item."""
    if kind not in {"image", "video"}:
        raise ValueError(f"unsupported media kind: {kind!r}")
    if not isinstance(source, str) or not source:
        raise ValueError("media source must be a non-empty string")
    return {"kind": kind, "source": source}


def media_references_digest(references: Iterable[dict[str, str]]) -> str | None:
    """Hash canonical ordered references without exposing their URLs to prompts."""
    normalized = [
        canonical_media_reference(item["kind"], item["source"])
        for item in references
    ]
    if not normalized:
        return None
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "DEFAULT_MAX_BYTES",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_TIMEOUT",
    "MediaDecodeError",
    "MediaError",
    "MediaPayload",
    "MediaSecurityError",
    "MediaTooLargeError",
    "VideoFrame",
    "canonical_media_reference",
    "decode_image",
    "decode_video",
    "load_media",
    "media_references_digest",
]

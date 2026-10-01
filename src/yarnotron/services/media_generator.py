from __future__ import annotations

import html
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import Depends

from yarnotron.core.config import Settings, get_settings
from yarnotron.models.media import MediaAsset


@dataclass(frozen=True)
class GeneratedMedia:
    asset_path: str
    asset_url: str
    mime_type: str
    generator: str
    revised_prompt: str


class MediaGenerator(Protocol):
    async def generate(self, asset: MediaAsset) -> GeneratedMedia: ...


class LocalSvgPlaceholderGenerator:
    name = "local_svg_placeholder"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def generate(self, asset: MediaAsset) -> GeneratedMedia:
        output_dir = Path(self._settings.media_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{asset.id}.svg"
        output_path = output_dir / filename
        revised_prompt = _build_child_safe_prompt(asset.prompt)
        output_path.write_text(_render_svg(revised_prompt), encoding="utf-8")
        return GeneratedMedia(
            asset_path=str(output_path),
            asset_url=f"/media/{filename}",
            mime_type="image/svg+xml",
            generator=self.name,
            revised_prompt=revised_prompt,
        )


def get_media_generator(settings: Settings = Depends(get_settings)) -> MediaGenerator:
    return LocalSvgPlaceholderGenerator(settings)


def _build_child_safe_prompt(prompt: str) -> str:
    return (
        "Child-safe storybook illustration, gentle and non-scary, no gore, no explicit content: "
        f"{prompt.strip()}"
    )


def _render_svg(prompt: str) -> str:
    escaped_prompt = html.escape(prompt)
    lines = textwrap.wrap(prompt, width=46)[:7]
    text_lines = "\n".join(
        f'<text x="64" y="{196 + index * 30}" class="prompt">{html.escape(line)}</text>'
        for index, line in enumerate(lines)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1024" height="768" viewBox="0 0 1024 768">
  <defs>
    <linearGradient id="sky" x1="0" x2="1" y1="0" y2="1">
      <stop offset="0" stop-color="#7cb7b6"/>
      <stop offset="0.52" stop-color="#f2d28b"/>
      <stop offset="1" stop-color="#f7f3df"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="8" stdDeviation="10" flood-color="#2f3b37" flood-opacity="0.2"/>
    </filter>
    <style>
      .title {{ font: 700 40px ui-sans-serif, system-ui, sans-serif; fill: #24312f; }}
      .prompt {{ font: 500 24px ui-sans-serif, system-ui, sans-serif; fill: #24312f; }}
      .caption {{ font: 500 18px ui-sans-serif, system-ui, sans-serif; fill: #5c675f; }}
    </style>
  </defs>
  <rect width="1024" height="768" fill="url(#sky)"/>
  <circle cx="812" cy="138" r="74" fill="#fff4bd" opacity="0.9"/>
  <path d="M0 606 C164 512 276 612 432 540 C596 466 764 500 1024 418 L1024 768 L0 768 Z" fill="#4f867d"/>
  <path d="M0 670 C190 596 348 650 512 602 C700 548 812 590 1024 524 L1024 768 L0 768 Z" fill="#2f625b"/>
  <g filter="url(#shadow)">
    <rect x="48" y="82" width="620" height="356" rx="8" fill="#fffaf0" opacity="0.94"/>
    <text x="64" y="142" class="title">Approved Story Image</text>
    {text_lines}
    <text x="64" y="402" class="caption">Placeholder generator: ready to swap for a local image model.</text>
  </g>
  <title>{escaped_prompt}</title>
</svg>
"""

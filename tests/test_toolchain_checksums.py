"""Toolchain downloads are checked against recorded sha256s. Hermetic: the
"download" is a file:// URL."""

from __future__ import annotations

import hashlib

import pytest

from train import toolchain


def _payload(tmp_path):
    src = tmp_path / "archive.tar.gz"
    src.write_bytes(b"not really a toolchain")
    return src, hashlib.sha256(src.read_bytes()).hexdigest()


def test_a_matching_download_is_kept(tmp_path):
    src, sha = _payload(tmp_path)
    dest = tmp_path / "got.tar.gz"
    toolchain.download_verified(src.as_uri(), dest, sha)
    assert dest.read_bytes() == src.read_bytes()


def test_a_mismatched_download_is_refused_and_deleted(tmp_path):
    src, _ = _payload(tmp_path)
    dest = tmp_path / "got.tar.gz"
    with pytest.raises(SystemExit, match="checksum mismatch"):
        toolchain.download_verified(src.as_uri(), dest, "0" * 64)
    assert not dest.exists()


def test_every_pinned_asset_has_a_checksum():
    for arch in ("arm64", "x86_64"):
        for os_name in ("linux", "macos"):
            asset = f"wasi-sdk-{toolchain.WASI_SDK_VERSION}.0-{arch}-{os_name}"
            assert asset in toolchain.WASI_SDK_SHA256, asset
    for arch in ("aarch64", "x86_64"):
        for os_name in ("linux", "macos"):
            asset = f"wit-bindgen-{toolchain.WIT_BINDGEN_VERSION}-{arch}-{os_name}.tar.gz"
            assert asset in toolchain.WIT_BINDGEN_SHA256, asset
    for sha in [*toolchain.WASI_SDK_SHA256.values(), *toolchain.WIT_BINDGEN_SHA256.values()]:
        assert len(sha) == 64 and int(sha, 16) >= 0

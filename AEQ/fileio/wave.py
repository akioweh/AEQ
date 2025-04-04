"""
Utilities for reading and writing WAV files
"""


import struct
from collections.abc import Iterable
from dataclasses import dataclass, astuple
from io import SEEK_CUR
from itertools import zip_longest
from typing import Final, TypeAlias, Literal
from warnings import warn

from numpy import float64, int32
from numpy.typing import NDArray

SampleRate: TypeAlias = Literal[8000, 11025, 16000, 22050, 32000, 44100, 48000, 96000, 192000, 384000]
BitDepth: TypeAlias = Literal[8, 16, 24, 32, 64]  # 64 is for floating point samples only
SampleType: TypeAlias = int | float
SampleType_np: TypeAlias = int32 | float64
Channel: TypeAlias = list[SampleType] | NDArray[SampleType_np]

INT_TYPES: Final[dict[int, str]] = {
    8: 'b',
    16: 'h',
    # 24 is not supported by struct
    32: 'i'
}

FLOAT_TYPES: Final[dict[int, str]] = {
    32: 'f',
    64: 'd'
}

ENCODINGS: Final[dict[str, str]] = {
    'PCM_8': 'b',
    'PCM_16': 'h',
    'PCM_24': 'i',
    'PCM_32': 'i',
    'FLOAT_32': 'f',
    'FLOAT_64': 'd'
}


@dataclass
class WaveFormat:
    fmt_code: int
    channels: int
    sample_rate: int
    bit_depth: BitDepth
    encoding: str

    def __iter__(self):
        # noinspection PyTypeChecker
        return iter(astuple(self))


def create_format(
        channel_count: int = 1,
        sample_rate: SampleRate | int = 44100,
        bit_depth: BitDepth = 16,
        is_float: bool = False
) -> WaveFormat:
    """Create a WaveFormat object."""

    assert channel_count > 0, 'Channel count must be positive'

    if is_float:
        assert bit_depth in (32, 64), 'Invalid bit depth for float format'
        encoding = f'FLOAT_{bit_depth}'
    else:
        assert bit_depth in (8, 16, 24, 32), 'Invalid bit depth for integer format'
        encoding = f'PCM_{bit_depth}'

    return WaveFormat(
        fmt_code=3 if is_float else 1,
        channels=channel_count,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        encoding=encoding
    )


def parse_riff(file: str) -> tuple[bytes, WaveFormat, dict | None]:
    """Parse a RIFF WAVE file by RIFF chunks
    and return the format, audio data, and any extra data (from unknown chunks)
    """

    with open(file, 'rb') as f:
        riff_tag, size, wave_tag = struct.unpack('<4sI4s', f.read(12))
        assert riff_tag == b'RIFF', 'Not a RIFF file'
        assert wave_tag == b'WAVE', 'Not a WAVE file'

        extra_data = {}

        read_size = 4  # 4 is already read as b'WAVE'
        next_chunk = f.read(8)
        seen_fmt = False
        seen_data = False
        while next_chunk:
            assert len(next_chunk) == 8, 'Unexpected end of file'
            read_size += 8
            chunk_tag, chunk_size = struct.unpack('<4sI', next_chunk)

            if chunk_tag == b'INFO':
                f.seek(chunk_size, SEEK_CUR)  # skip INFO chunk
                read_size += chunk_size

            elif chunk_tag == b'fmt ':
                assert not seen_fmt, 'Multiple fmt chunks detected'
                seen_fmt = True
                assert chunk_size in (16, 18, 40), f'Unexpected chunk size: {chunk_size}. Expected 16, 18, or 40'

                format_tag, n_channels, sample_rate, byte_rate, block_align, bit_depth = struct.unpack('<HHIIHH', f.read(16))
                read_size += 16
                assert byte_rate == sample_rate * n_channels * bit_depth // 8, 'Incorrect byte rate'
                assert block_align == n_channels * bit_depth // 8, 'Incorrect block align'

                if chunk_size > 16:  # read extra data for non-PCM formats (fmt extension)
                    fmt_ext_size = struct.unpack('<H', f.read(2))[0]
                    read_size += 2
                    assert fmt_ext_size in (0, 22), 'Incorrect fmt extension size'

                    if fmt_ext_size == 22:
                        valid_bits_per_sample, channel_mask, sub_format = struct.unpack('<HH16s', f.read(20))
                        read_size += 20
                        warn(f'RIFF fmt extension block detected. '
                             f'Extensible format is not fully supported. '
                             f'Ignoring extra data: {valid_bits_per_sample}, {channel_mask}, {sub_format}')
                        assert valid_bits_per_sample <= bit_depth, (f'unexpected valid bits per sample: {valid_bits_per_sample}'
                                                                    f'against bit depth: {bit_depth}')

            elif chunk_tag == b'fact':
                assert chunk_size == 4, f'Unexpected fact chunk size: {chunk_size}. Expected 4'
                f.seek(4, SEEK_CUR)  # skip fact chunk
                read_size += 4

            elif chunk_tag == b'data':
                assert not seen_data, 'Multiple data chunks detected'
                seen_data = True
                data = f.read(chunk_size)
                read_size += chunk_size
                if chunk_size % 2:
                    f.seek(1, SEEK_CUR)  # skip padding byte
                    read_size += 1

            else:
                unknown = f.read(chunk_size)
                read_size += chunk_size
                extra_data[chunk_tag] = unknown

            next_chunk = f.read(8)

        assert seen_fmt, 'No fmt chunk found'
        assert seen_data, 'No data chunk found'
        assert read_size == size, 'File size mismatch'

        if format_tag == 1:
            encoding = f'PCM_{bit_depth}'
        elif format_tag == 3:
            encoding = f'FLOAT_{bit_depth}'
        else:
            raise ValueError(f'Unsupported encoding: {format_tag:04X}')
        format_info = WaveFormat(format_tag, n_channels, sample_rate, bit_depth, encoding)

        if not extra_data:
            extra_data = None
        return data, format_info, extra_data


def parse_data(data: bytes, format_info: WaveFormat) -> Channel | tuple[Channel]:
    """Parse raw audio data into a list of samples based on the format info

    if more than one channel, return a tuple of lists of samples for each channel
    """

    enc_format, channels, sample_rate, bit_depth, encoding = format_info
    data_len = len(data)
    n_frames = data_len // (bit_depth // 8)

    if enc_format == 1:  # int-PCM
        if bit_depth == 24:
            # not a valid C-type, so we have to unpack manually
            samples = [
                int.from_bytes(data[i:i + 3], 'little', signed=True)
                for i in range(0, data_len, 3)
            ]
        else:
            if bit_depth not in INT_TYPES:
                raise ValueError(f'Unsupported bit depth: {bit_depth}')
            ctype = INT_TYPES[bit_depth]
            samples = list(struct.unpack(f'<{n_frames}{ctype}', data))

    elif enc_format == 3:  # float-PCM
        if bit_depth not in FLOAT_TYPES:
            raise ValueError(f'Unsupported floating bit depth: {bit_depth}')
        ctype = FLOAT_TYPES[bit_depth]
        samples = list(struct.unpack(f'<{n_frames}{ctype}', data))

    else:
        raise ValueError(f'Unsupported encoding: {enc_format}')

    # de-interleave channels
    if channels > 1:
        samples = tuple(samples[i::channels] for i in range(channels))

    return samples


def parse_wave(file: str) -> tuple[Channel | tuple[Channel], WaveFormat]:
    """Parse a RIFF WAVE file and return the format and audio data"""
    data, format_info, _ = parse_riff(file)
    samples = parse_data(data, format_info)
    return samples, format_info


def write_wave(file: str, tracks: Channel | list[Channel], format_info: WaveFormat):
    """Write a RIFF WAVE file with the given format and samples

    sample format should match the format info (denormalize ints, normalize floats)
    if more than one channel, samples should be a tuple of lists of samples for each channel
    """

    if not isinstance(tracks, Iterable):
        tracks = [tracks]
    # tracks = [list(i) for i in tracks]  # make sure channels are lists and not np arrays

    data = encode_data(tracks, format_info)
    write_riff(file, data, format_info)


def encode_data(samples: Channel | list[Channel], format_info: WaveFormat) -> bytes:
    """Encode samples into raw audio data based on the format info

    if more than one channel, samples should be a tuple of lists of samples for each channel
    """

    enc_format, n_channels, sample_rate, bit_depth, encoding = format_info

    if isinstance(samples, Iterable) and isinstance(samples[0], Iterable):
        assert n_channels == len(samples), 'Channel count mismatch'
        filler = 0.0 if enc_format == 3 else 0
        # noinspection PyArgumentList
        samples_ = [
            frame
            for channel in zip_longest(*samples, fillvalue=filler)
            for frame in channel
        ]
    else:
        assert n_channels == 1, 'Single channel data must be a list'
        samples_ = samples

    if enc_format == 1:  # int-PCM
        if bit_depth == 24:
            # not a valid C-type, so we have to pack manually
            data = b''.join(
                sample.to_bytes(3, 'little', signed=True)
                for sample in samples_
            )
        else:
            if bit_depth not in INT_TYPES:
                raise ValueError(f'Unsupported bit depth: {bit_depth}')
            ctype = INT_TYPES[bit_depth]
            data = struct.pack(f'<{len(samples_)}{ctype}', *samples_)

    elif enc_format == 3:  # float-PCM
        if bit_depth not in FLOAT_TYPES:
            raise ValueError(f'Unsupported floating bit depth: {bit_depth}')
        ctype = FLOAT_TYPES[bit_depth]
        data = struct.pack(f'<{len(samples_)}{ctype}', *samples_)

    else:
        raise ValueError(f'Unsupported encoding: {enc_format}')

    return data


def write_riff(file: str, data: bytes, format_info: WaveFormat, extra_data: dict | None = None):
    """Write a RIFF WAVE file with the given format, audio data, and extra data"""

    enc_format, channels, sample_rate, bit_depth, encoding = format_info

    write_buffer: list[bytes] = [
        b'RIFF',
        b'\x00\x00\x00\x00',  # file size placeholder
        b'WAVE',
        b'fmt ',
        struct.pack('<I', 16)
    ]

    # fmt chunk
    byte_rate = sample_rate * bit_depth * channels // 8
    block_align = bit_depth * channels // 8
    fmt = (enc_format, channels, sample_rate, byte_rate, block_align, bit_depth)
    write_buffer.append(struct.pack('<HHIIHH', *fmt))

    # fact chunk
    if enc_format == 3:  # float-PCM
        fact = 4, len(data) // (bit_depth // 8)
        write_buffer.extend((
            b'fact',
            struct.pack('<II', *fact)
        ))

    # extra data
    if extra_data:
        for key, value in extra_data.items():
            assert isinstance(key, bytes), 'Extra data key must be bytes'
            assert isinstance(value, bytes), 'Extra data value must be bytes'
            write_buffer.extend((
                key,
                struct.pack('<I', len(value)),
                value
            ))

    # data chunk
    write_buffer.extend((
        b'data',
        struct.pack('<I', len(data)),
        data
    ))

    # fill in file size
    file_size = sum(len(chunk) for chunk in write_buffer) - 8
    write_buffer[1] = struct.pack('<I', file_size)

    with open(file, 'wb') as wf:
        wf.write(b''.join(write_buffer))

"""
FluxPack: Schema-free, Shannon-optimal serialisation for ML pipelines.

Faster and smaller than JSON. Built for ML engineers.

Quick Start:
    >>> import fluxpack
    >>> data = {"epoch": 1, "loss": 0.5, "accuracy": 0.8}
    >>> encoded = fluxpack.encode(data)
    >>> decoded = fluxpack.decode(encoded)
    >>> assert decoded == data

Batch Processing:
    >>> messages = [{"epoch": i, "loss": 2.5 - i * 0.1} for i in range(100)]
    >>> encoded = fluxpack.encode_batch(messages)
    >>> decoded = fluxpack.decode_batch(encoded)

Streaming:
    >>> writer = fluxpack.StreamWriter()
    >>> for msg in messages:
    ...     writer.write_data(msg)
    >>> bytes = writer.finish()

Compression:
    >>> compressed = fluxpack.encode_compressed(data, level=3)
    >>> data = fluxpack.decode_compressed(compressed)

Stateful Encoder/Decoder:
    >>> encoder = fluxpack.Encoder()
    >>> for batch in training_batches:
    ...     encoded = encoder.encode(batch)
"""

from fluxpack._fluxpack import (
    encode,
    decode,
    encode_batch,
    decode_batch,
    encode_compressed,
    decode_compressed,
    Encoder,
    Decoder,
    StreamWriter,
    StreamReader,
    __version__,
)

__all__ = [
    "encode",
    "decode",
    "encode_batch",
    "decode_batch",
    "encode_compressed",
    "decode_compressed",
    "Encoder",
    "Decoder",
    "StreamWriter",
    "StreamReader",
    "__version__",
]

# ResilientFile

**A pure-Python Reed–Solomon library for file protection, error correction, and recovery.**

ResilientFile implements Reed–Solomon error-correcting codes over GF(256), with support for correcting corrupted bytes, reconstructing known erasures, and recovering protected files.

The project also includes CRC32-protected packet framing, packet interleaving, and a Gilbert–Elliott channel simulator for evaluating recovery over unreliable communication links.

## Features

- Reed–Solomon encoding and decoding over GF(256)
- Configurable RS(n, k) parameters, with n ≤ 255
- Systematic encoding
- Unknown-byte-error correction
- Known-erasure recovery
- Joint error-and-erasure correction
- File protection and recovery with SHA-256 integrity verification
- CRC32 packet framing and corruption detection
- Packet interleaving for burst-loss experiments
- Gilbert–Elliott channel simulation
- Command-line interface
- Automated test suite

## Installation

ResilientFile can be installed from a built Python wheel:

```bash
python -m pip install resilientfile-0.1.0-py3-none-any.whl
```

A public PyPI release is planned.

## Quick Start

### Python API

```python
import resilientfile as rf

original = b"HELLO"

encoded = rf.encode(original, n=9, k=5)

damaged = bytearray(encoded)
damaged[0] ^= 17
damaged[8] ^= 29

result = rf.decode(bytes(damaged), n=9, k=5)

assert result.data == original

print("Recovered:", result.data)
print("Corrected positions:", result.corrected_positions)
```

### Protect and Recover Files

```python
import resilientfile as rf

original = b"Important data that must be protected."

protected = rf.protect_bytes(original)

recovered = rf.recover_bytes(protected)

assert recovered == original
```

### Command-Line Interface

Protect a file:

```bash
resilientfile protect original.pdf protected.rf
```

Recover it:

```bash
resilientfile recover protected.rf recovered.pdf
```

Configure the Reed–Solomon parameters:

```bash
resilientfile protect original.pdf protected.rf --n 255 --k 223
```

## How Reed–Solomon Recovery Works

For an RS(n, k) code:

- `k` is the number of original data symbols.
- `n` is the total number of encoded symbols.
- `n-k` is the number of parity symbols.

With `r = n-k`, the correction guarantee for a codeword is:

**2e + s ≤ r**

Where:

- `e` is the number of unknown erroneous symbols.
- `s` is the number of known erased symbols.

For RS(255,223), 32 parity symbols can correct up to 16 unknown byte errors, 32 known erasures, or supported combinations of both within that bound.

## Experimental Evaluation

ResilientFile includes a packet-level RS(40,32) simulation using 200-byte payloads.

In one benchmark of 50 trials per scenario and transmission mode, using a synthetic 19,200-byte input:

| Scenario | Sequential recovery | Interleaved recovery |
|---|---:|---:|
| Mild burst loss | 100% | 100% |
| Moderate burst loss | 76% | 98% |
| Severe burst loss | 0% | 4% |

Under the moderate-loss configuration, interleaving improved the observed complete-file recovery rate by 22 percentage points.

These results are specific to the configured channel model, random seeds, corruption probabilities, input size, and coding parameters. They are not universal recovery guarantees.

## Testing

The project has passed 159 automated tests in the current development environment.

Run the tests from the repository root:

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

## Limitations

- File-container metadata currently requires an intact header.
- Recovery is bounded by the configured Reed–Solomon redundancy.
- CRC32 detects accidental corruption but is not a cryptographic authentication mechanism.
- SHA-256 verifies recovered content against the digest stored by the container; it does not authenticate the source.
- Packet experiments are simulations and do not yet establish performance on physical radio hardware.
- The implementation has not undergone an independent security or reliability audit.

## Project Status

**Version:** 0.1.0

**Stage:** Working Python library and experimental prototype.

Planned improvements include more resilient container metadata, expanded benchmarking, stronger error reporting, and integration with offline or intermittently connected communication applications.

## License

ResilientFile is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
"""
## ======================================================== ## 
sync_client_hashing_logic.py
## ======================================================== ## 

The hashing logic of the sync client split into (3) foundational parts

1) 
A rolling hash ( aka the "gear hash" as per the FastCDC paper) which decides where a file would be cut into chunks.
Primarily job is to create chunk boundaries based on the content rather than byte offsets; so an edit in (1) place doesn't shift every subsequent boundary

2) 
A SHA-256 digest of each chunk. This is the chunk's identity as the server keys its content-defined storage on it
The client requests which digest (chunk) is missing; the server refuses any chunk whose bytes do not hash to the digest it was sent under.

3) 
A SHA-256 digest of the whole file -> computed in the same single pass over the bytes. 
The server recomputes it after assembling the chunks and both sides compare (no inherent trust); thus, this forms the integrity guarantee

"""

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Callable, Iterable


## ======================================================== ## 
## rolling hash: to decide where chunks will begin & end          
# init of the gear hash requires one pseudo-random 64-bit number for each of the 256 possible byte values. 
# the table is generated from a [fixed] seed such that every client (or version of it) builds exactly the same table
# AKA changing the seed = every chunk boundary moves.
## ======================================================== ## 

# fixed seed; 64bit bitmask of 64(1) = max value for unsigned 64-bit integer 
GEAR_TABLE_SEED = 0x5EED_CDC
SIXTY_FOUR_BIT_MASK = (1 << 64) - 1


def build_gear_table(
        seed: int = GEAR_TABLE_SEED
        ) -> tuple[int, ...]:
    
    """Return 256 pseudo-random 64-bit integers, one per possible byte value."""
    generator = random.Random(seed)
    return tuple(generator.getrandbits(64) for _ in range(256))

GEAR_TABLE = build_gear_table()


## ======================================================== ## 
## the boundary masks
# a boundary is formed at a byte when: (rolling_hash & mask) == 0. 
# see write-up for logic for normalised chunking to pull chunk sizes back towards a target value via asymmetry
## ======================================================== ## 

def build_boundary_mask(
        number_of_one_bits: int
        ) -> int:
    """Return a 64bit mask with the given number of (1) bits.
    the gear hash shifts left on every byte, 
    so the high bits summarise a longer window of recent bytes; 
    spreading the mask over them makes the boundary decision depend on more context,
    which gives more stable boundaries. The spread is deterministic (fixed seed)
    """
    generator = random.Random(0xA5A5 + number_of_one_bits)
    chosen_bit_positions = generator.sample(range(16, 64), number_of_one_bits)

    mask = 0
    for position in chosen_bit_positions:
        mask |= 1 << position

    return mask

# use an immutable/read-only dataclass 
@dataclass(frozen=True)
class ChunkingPolicy:
    """The three sizes that shape our chunks + the masks derived from them

    :minimum_size =  no boundary is allowed before this many bytes. Keeps chunks from becoming tiny (each chunk costs a 32-byte digest and, when sent, one request).
    :target_size = the average/target chunk size 
    :maximum_size = a boundary is forced here even if the rolling hash never matched (for example inside long runs of identical bytes). Bounds memory use and the cost of losing one chunk to a network interruption.
    """

    minimum_size: int = 16 * 1024
    target_size: int = 64 * 1024
    maximum_size: int = 256 * 1024

    def __post_init__(self) -> None:
        if not (0 < self.minimum_size <= self.target_size <= self.maximum_size):
            raise ValueError(
                "chunk sizes must satisfy restraint: 0 < minimum <= target <= maximum"
                )

    @property
    def strict_mask(self) -> int:
        """used while the chunk is smaller than the target -> is harder to match by design"""
        bits_for_target = self.target_size.bit_length() - 1   
        return build_boundary_mask(min(bits_for_target + 2, 40))

    @property
    def loose_mask(self) -> int:
        """used once the chunk is past the target -> easier to match by design"""
        bits_for_target = self.target_size.bit_length() - 1
        return build_boundary_mask(max(bits_for_target - 2, 1))


DEFAULT_POLICY = ChunkingPolicy()


def find_chunk_boundary(
        buffer: bytes, 
        policy: ChunkingPolicy = DEFAULT_POLICY
        ) -> int:
    """return the length of the first chunk that starts at buffer[0]

    The buffer must hold (at least) policy.maximum_size bytes; 
    unless the file has ended, in which case whatever remains is acceptable

    Reading order of the decision:

    1) If the buffer is not longer than the minimum size then the whole buffer is the chunk (this only happens at the end of a file)
    2) Otherwise, roll/move-forward the gear hash over the bytes. The first (minimum_size) bytes only warm the hash up - no boundary may be placed here
    3) From the minimum size to the target size, cut where the (stricter) mask matches; 
    -- else past the target size, cut where the (looser) mask matches
    4) Else, If nothing matched by the maximum size - > cut there

    This loop is the path of the whole client.
    """
    furthest_we_may_look = min(len(buffer), policy.maximum_size)
    if furthest_we_may_look <= policy.minimum_size:
        return furthest_we_may_look

    strict_mask, loose_mask = policy.strict_mask, policy.loose_mask  # once, not per byte

    rolling_hash = 0
    for position in range(policy.minimum_size):
        rolling_hash = _roll(rolling_hash, buffer[position])

    for position in range(policy.minimum_size, furthest_we_may_look):
        rolling_hash = _roll(rolling_hash, buffer[position])
        mask = strict_mask if position < policy.target_size else loose_mask
        if rolling_hash & mask == 0:
            return position + 1                # the boundary sits AFTER this byte

    return furthest_we_may_look                # forced cut at the maximum size


def _roll(rolling_hash: int, byte_value: int) -> int:
    """move-forward/advance the gear hash by (1) byte: shift left + add the byte's gear value"""
    return ((rolling_hash << 1) + GEAR_TABLE[byte_value]) & SIXTY_FOUR_BIT_MASK


## ======================================================== ## 
## the cryptographic digests: chunk identity & whole-file integrity
## ======================================================== ## 

def sha256_hex(data: bytes) -> str:
    """The digest used everywhere as 64 lowercase hexadecimal characters"""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Chunk:
    """ one chunk of a file: where it is / how long it is / what it is called

    the digest = the chunk's identity. 
    Two chunks with the same digest have the same bytes, wherever they occur - in this file, even another file, or a file on another client.
    """

    offset: int
    length: int
    digest: str


@dataclass(frozen=True)
class FileFingerprint:
    """aggregate of everything we gleam from one pass over a file.

    chunks in file order; sending their digests to the server is how discovery of which bytes are still required occurs
    file_digest = SHA-256 of the whole file; the server must arrive at the same value after assembling the chunks.
    size = total bytes; a cheap extra check alongside the digest although technically not overly reliable given miniture changes or a + and - may not move the dial
    """

    size: int
    file_digest: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def chunk_digests(self) -> list[str]:
        return [chunk.digest for chunk in self.chunks]


def fingerprint_stream(
        read: Callable[[int], bytes],
        policy: ChunkingPolicy = DEFAULT_POLICY
        ) -> FileFingerprint:
    """both chunk and hash a stream of bytes in a cruicially (single) pass.
    read(n) must behave like file.read(n); return up to (n) bytes, or (b) at the end. 
    The function keeps only about one maximum-size chunk in memory at a time, so it handles files of any size
    The whole-file digest is fed the same bytes as the chunker as they go past, which is why the file is read from disk (exactly) once.
    """
    whole_file_hasher = hashlib.sha256()
    chunks: list[Chunk] = []
    buffer = bytearray()
    offset_of_buffer_start = 0
    end_of_stream = False

    while True:
        # Keep enough bytes buffered that a chunk never straddles a read
        while not end_of_stream and len(buffer) < policy.maximum_size:
            more = read(policy.maximum_size)
            if not more:
                end_of_stream = True
            else:
                buffer.extend(more)

        if not buffer:
            break                                   # nothing left / done

        chunk_length = find_chunk_boundary(bytes(buffer), policy)
        chunk_bytes = bytes(buffer[:chunk_length])

        whole_file_hasher.update(chunk_bytes)
        chunks.append(Chunk(offset=offset_of_buffer_start,
                            length=chunk_length,
                            digest=sha256_hex(chunk_bytes)))

        del buffer[:chunk_length]
        offset_of_buffer_start += chunk_length

    return FileFingerprint(
                size=offset_of_buffer_start,
                file_digest=whole_file_hasher.hexdigest(),
                chunks=chunks
                )


def fingerprint_file(
        path: Path | str,
        policy: ChunkingPolicy = DEFAULT_POLICY
        ) -> FileFingerprint:
    
    """convenience wrapper: -> func (fingerprint_stream) over a file on disk"""
    with open(path, "rb") as file_handle:
        return fingerprint_stream(file_handle.read, policy)


def read_chunk(
        file_handle: BinaryIO, 
        chunk: Chunk
        ) -> bytes:
    
    """read one chunk's bytes back from an open file for upload"""
    file_handle.seek(chunk.offset)
    return file_handle.read(chunk.length)


## ======================================================== ##
#  Verification -> the checks both client and server perform 
## ======================================================== ##

def chunk_bytes_match(
        data: bytes, 
        expected_digest: str
        ) -> bool:
    
    """checking: 'do these bytes really compute to this digest?'

    AKA the client asks this as it re-reads each chunk for upload 
    The server requests it before storing any chunk (a mismatch means corruption in transit & the chunk is rejected).
    """
    return sha256_hex(data) == expected_digest


def assembled_file_matches(chunks_in_order: Iterable[bytes],
                           expected_digest: str,
                           expected_size: int) -> bool:
    """checking: would assembling/re-concatenating these chunks reproduce the original file exactly?

    This is the server side's at commit time check, written so that it never needs the whole file in memory -> the chunks are streamed through one hasher
    It catches everything the per-chunk check cannot; e.g. a missing chunk / chunks assembled in the wrong order.
    """
    hasher = hashlib.sha256()
    total_size = 0
    for chunk_bytes in chunks_in_order:
        hasher.update(chunk_bytes)
        total_size += len(chunk_bytes)
    return total_size == expected_size and hasher.hexdigest() == expected_digest


def digests_the_server_lacks(
        our_digests: Iterable[str],
        digests_the_server_has: set[str]
        ) -> list[str]:
    
    """checking: which of our chunks must actually be sent? (order preserved (sets) / no repeats)

    fundamentally, it is the question the digests exist to answer; the client asks the server & then this function is what the server computes.
    """
    already_listed: set[str] = set()
    needed: list[str] = []
    for digest in our_digests:
        if digest not in digests_the_server_has and digest not in already_listed:
            needed.append(digest)
            already_listed.add(digest)
    return needed




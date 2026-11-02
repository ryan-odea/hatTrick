# hatTrick (HATRX) - Hadamard encoding/decoding for Time-Resolved Crystallography

hatTrick (HATRX) is a lightweight package aimed at making the work of encoding and decoding the hadamard transform simple. This package is primarily aimed at time resolved serial crystallography (TRSC) and seeks to improve signal to noise ratio and indexing rates within the data provided. 

## Installation

This package is available through pypi and downloadable with the pip package manager
```shell
pip install hatTrick
```
## Dependencies

- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- h5py >= 3.0.0
- bitshuffle >= 0.3.0
- click >= 8.0.0

## Example Usage (With Crystfel)

While this package does support access through the python api, it is more simple to build a pipeline along the command line. This makes interacting with crystfel simpler as you can immediately place everything into a job script without the need for an intermediate python script.

For our example, let's assume we have 45 frames, each of 5ms, and want to do a rank 3 encoding of our data.

### (1) Encoding Data
This step intakes a list file containing information of where to find your .h5 data as well as information about your data (how many frames you would like to encode, how many frames exist in your data).

```bash
LIST_FILE={DATA.LST}
N_MERGED_FRAMES=3           # Must be prime and ≡ 3 (mod 4): 3, 7, 11, 19, 23, 31...
N_FRAMES=45
DATA_LOCATION="entry/data"
DATA_NAME="data"
OUTDIR="hadamard_outputs"
N_FILES=8                   # Process 8 files in parallel
WORKERS_PER_MERGE=4         # Workers per single merge

export LIST_FILE N_MERGED_FRAMES DATA_LOCATION DATA_NAME OUTDIR WORKERS_PER_MERGE

mkdir -p "$OUTDIR" logs

run_hadamard_merge() {
    local INPUT_FILE="$1"
    local BASENAME=$(basename "$INPUT_FILE" .h5)
    local OUTPUT_FILE="${OUTDIR}/${BASENAME}_hadamard.h5"
    
    hatrx encode \
        -f "$INPUT_FILE" \
        -o "$OUTPUT_FILE" \
        --n-frames "$N_FRAMES" \
        --n-merged-frames "$N_MERGED_FRAMES" \
        --type hadamard \
        --data-location "$DATA_LOCATION" \
        --data-name "$DATA_NAME" \
        --n-workers "$WORKERS_PER_MERGE"
}

export -f run_hadamard_merge

cat "$LIST_FILE" | parallel -j "$N_FILES" run_hadamard_merge {}
```

### (2) Indexing, Resolving Ambiguities, Scaling with Crystfel

As normal, you will index, resolve any ambiguities, and scale. Here, we provide this process with Crystfel, noting that you would need to change indexing parameters to those which best fits your data. Anecdotally, hadamard encoded files preform best with a higher SNR compared to your standard data processing hyperparameters.

For our example, let's assume a threshold of 8, SNR of 4.0, min-peaks of 8, and min-pix of 2 were the optimal hyperparameters found. In some of our data, we've found that each of the encoded files arrives at the same optimal hyperparameters; however, it's probably wisest to test each of the sets.

```shell
# 110 Encoded Files
indexamajig -g {.GEOM} -i {LIST_110} -o {110.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# 101 Encoded Files
indexamajig -g {.GEOM} -i {LIST_101} -o {101.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# 011 Encoded Files
indexamajig -g {.GEOM} -i {LIST_011} -o {011.STREAM} --indexing=xgandalf,asdf,mosflm,taketwo --peaks=peakfinder8 --int-radius=3,4,7 --multi --no-check-peaks --threshold=8 --min-snr=4.0 --min-peaks=8 --min-pix-count=2 -p {.CELL}

# Combine Streams
cat 110.STREAM 101.STREAM 011.STREAM > data.stream

# Resolve ambiguities together
ambigator -o {AMBI.STREAM} -w 6/mmm --lowres=10.0 --highres=3.0 --ncorr=1000 -j 32 --symmetry=6/m data.stream

# Scale (Custom output split with lists created before x n_frames)
partialator -o data_110 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_110}

partialator -o data_101 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_101}

partialator -o data_011 -y 6/m -i {AMBI.STREAM} --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split={LIST_011}
```

### (3) Decoding

Now that we have our encoded data in hkl format, with ambiguities resolved and appropriately scaled together, we can decode the data to retrieve our normal hkls.

```shell
hatrx decode -n 3 -p "data_110-*.hkl" -p "data_101-*.hkl" -p "data_011-*.hkl" -o .
```
### (3b) Decoding on the stream (recommended)

Scaling is not linear in the encoded intensities: partialator fits a scale, B factor and partiality per encoded chunk, so decoding after scaling mixes those factors into the result and can absorb real frame-to-frame changes. Decoding the stream instead inverts the raw integrated intensities, after which partialator scales ordinary single-frame chunks.

```shell
# Index and resolve ambiguities as above, then decode the ambigator output
hatrx decode-stream -n 3 -i {AMBI.STREAM} -o decoded.stream --symmetry 6/m --min-common 10

# One dataset per time point: block "<stem>_<bunch>", frame f  ->  t = n*bunch + f
awk '/^Image filename:/{fn=$3} /^Event:/{ev=$2; sub("//","",ev); k=split(fn,a,"_"); print fn, "//" ev, 3*a[k]+ev}' decoded.stream > split_t.lst

# Scale and merge the decoded frames as normal data
partialator -o decoded -y 6/m -i decoded.stream --model=xsphere --iterations=1 --push-res=1.5 -j 32 --custom-split=split_t.lst
```

The n chunks of a block are indexed independently, and a lattice can be indexed in any setting equivalent under its Laue group. ambigator only resolves the merohedral ambiguity, so the same reflection can carry a different `h k l` in each chunk. Pass `--symmetry` (the merging Laue group; currently `6/m`) so each chunk is relabelled into the setting of the first pattern before the reflections are joined. Without it, on real data almost no reflections are common to all n chunks. Blocks where the operator choice is not decisive, or with fewer than `--min-common` common reflections, are skipped.

The split line above assumes the default `{stem}_{tag}.h5` gating, where the event of an encoded file is the bunch index; replace `3` with your `n`.

Chunks are gated into blocks of `n` encoded chunks, one per S-matrix row, and each block is inverted together. By default a block is read off the encoder's `{stem}_{tag}.h5` file names plus the event number. When files are named differently, pass a split file with one line per encoded chunk:

```
# image filename    event   block   pattern (row index or binary tag)
run_0001_a.h5       5       blk5    110
run_0001_b.h5       5       blk5    101
run_0001_c.h5       5       blk5    011
```

```shell
hatrx decode-stream -n 3 -i {AMBI.STREAM} -o decoded.stream --split gating.txt
```

Decoded chunks carry the block id as `Image filename` and the frame index as `Event`, so each frame is a distinct crystal for partialator. Blocks with an unindexed chunk, a missing pattern, more than one crystal in a chunk, an ambiguous setting or too few common reflections are skipped and counted separately. The input is read chunk by chunk and unindexed chunks are not kept, but the indexed chunks are held in memory until the end (about 130 GB for a 25 GB stream). Sigmas are propagated in quadrature in both decoders.

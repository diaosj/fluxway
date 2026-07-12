# Computation Graph Visualization

Fluxway exports computation graphs for Google Model Explorer without generating
or maintaining a custom user interface.

## Installation

```bash
pip install "fluxway[visualization]"
```

The visualization extra installs `ai-edge-model-explorer`.

## Export And View

```python
from fluxway import view_graph, visualize_graph

bundle = visualize_graph(graph, "/tmp/training-graph")
view_graph(bundle)
```

`view_graph` starts the Model Explorer server and blocks until it is stopped.
Export and viewing are separate so a headless training process never opens a
browser or starts an HTTP server implicitly.

The bundle contains exactly two files:

- `manifest.json` is Fluxway's immutable computation-graph fact document.
- `model_explorer.json` is a replaceable Model Explorer projection generated
  only from the manifest.

An existing output directory raises `FileExistsError`. Passing `overwrite=True`
replaces it only when both files exactly match a projection regenerated from the
validated manifest. Replacement restores the previous bundle if the directory
swap fails.

## Model Explorer Mapping

SPMDs are Model Explorer namespaces. Operators are op nodes. Selecting an SPMD
or operator shows its full identifiers, description, mesh, resources, classified
inputs and outputs, signature identifiers, and TUUID mappings.

Search Model Explorer attributes using the exact identifiers shown in Perfetto:

- `manifest_id`
- `spmd_id`
- `op_id`
- `function_id`
- `input_signature_id`
- `output_signature_id`

Model Explorer removes layers that contain one operator by default. Fluxway also
places the SPMD identity, class, description, mesh, and resources on every
operator, so those facts remain searchable and copyable. Enable Model Explorer's
`Keep layers with a single op node child` setting when every SPMD must remain a
visible layer in the overview.

Model Explorer may draw multiple transfers between the same operator pair as one
edge. Each transfer remains separate in operator input/output metadata and in
`manifest.json`, including `source output -> target input -> tuuid` and the
local, conditional, or symmetric properties.

## Remote Use

Bind the viewer to loopback and use SSH forwarding:

```python
view_graph(
    "/tmp/training-graph",
    host="127.0.0.1",
    port=8080,
    open_browser=False,
)
```

```bash
ssh -L 8080:127.0.0.1:8080 user@training-host
```

The bundle can also be copied to another machine and opened with:

```bash
model-explorer /path/to/training-graph/model_explorer.json
```

## Copy To A Mac

Copy the complete bundle so `manifest.json` remains beside the replaceable
Model Explorer projection. Run this on the Mac, replacing the SSH host and
remote path:

```bash
mkdir -p "$HOME/Downloads/training-graph"
rsync -av --progress \
  user@training-host:/tmp/training-graph/ \
  "$HOME/Downloads/training-graph/"
ls -1 "$HOME/Downloads/training-graph"
```

The final command must show exactly `manifest.json` and
`model_explorer.json`. For a one-time copy, `scp` is also sufficient:

```bash
scp -r user@training-host:/tmp/training-graph "$HOME/Downloads/"
```

Install Model Explorer in an isolated Mac environment and open the copied
projection on loopback:

```bash
brew install python@3.12  # only when python3.12 is not already installed
python3.12 -m venv "$HOME/.venvs/fluxway-model-explorer"
source "$HOME/.venvs/fluxway-model-explorer/bin/activate"
python -m pip install ai-edge-model-explorer
model-explorer \
  "$HOME/Downloads/training-graph/model_explorer.json" \
  --host 127.0.0.1 --port 8080
```

To compare with a trace, copy the `.pftrace` file to the Mac as well:

```bash
scp user@training-host:/tmp/recommendation-traces/run.pftrace \
  "$HOME/Downloads/run.pftrace"
open "https://ui.perfetto.dev"
```

Load `run.pftrace` from Perfetto's **Open trace file** action. Search the
`manifest_id`, `spmd_id`, `op_id`, `function_id`, and signature IDs shown in
Model Explorer to compare the two views.

## Future Viewers

`fluxway.graph_manifest(graph)` returns the same frozen
`ComputationGraphManifest` stored on the built graph. A future viewer consumes
this document rather than importing Builder, profiler, Model Explorer, or
runtime internals. Identity and transfer fields must never be reconstructed from
display names.

Model Explorer is desktop-oriented. It does not parse `.pftrace` files or
provide automatic navigation to Perfetto. The two tools correlate through
fields copied from the same manifest facts.

The official UI loaded and searched the 1k-operator proof, but a single SPMD
containing 10k operators and 9,999 transfers did not finish its initial layout
within 180 seconds. Bundle generation still succeeds; use the manifest directly
for automated inspection of graphs at that density.

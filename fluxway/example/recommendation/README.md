# Recommendation Example

Install the visualization dependency before exporting the computation graph:

```bash
pip install -e ".[visualization]"
```

Exporting is explicit and does not start Model Explorer or change the training
flow:

```bash
python -m fluxway.example.recommendation.train \
  --graph-visualization-dir /tmp/recommendation-graph
```

Use `--overwrite-graph-visualization` only to replace a directory previously
created by Fluxway. Other existing directories are rejected.

Open the exported graph locally:

```bash
model-explorer /tmp/recommendation-graph/model_explorer.json
```

Graph export and Perfetto tracing can be enabled together:

```bash
python -m fluxway.example.recommendation.train \
  --graph-visualization-dir /tmp/recommendation-graph \
  --profiler-trace-dir /tmp/recommendation-traces
```

On a remote machine, keep Model Explorer on loopback and forward the port:

```bash
model-explorer /tmp/recommendation-graph/model_explorer.json \
  --host 127.0.0.1 --port 8080 --no_open_in_browser
ssh -L 8080:127.0.0.1:8080 user@training-host
```

Alternatively, copy the bundle to a workstation that has Model Explorer
installed and open `model_explorer.json` there.

From a Mac, copy the entire recommendation bundle:

```bash
mkdir -p "$HOME/Downloads/recommendation-graph"
rsync -av --progress \
  user@training-host:/tmp/recommendation-graph/ \
  "$HOME/Downloads/recommendation-graph/"
```

Keep `manifest.json` beside `model_explorer.json`; it is the canonical source
for matching Model Explorer identifiers with the copied Perfetto trace. Follow
[Computation Graph Visualization](../../../docs/graph_visualization.md#copy-to-a-mac)
for the Mac Python environment, local Model Explorer server, `.pftrace` copy,
and Perfetto steps.

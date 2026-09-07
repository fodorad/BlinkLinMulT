# Demos

Everything you can run to *see* the models work. Each is a thin driver over the
package — the logic lives in `blinklinmult/`, so these stay readable.

| what | run it | needs |
|---|---|---|
| **Offline video** | `make video-demo` | nothing but the weights |
| **Live webcam** | `make webcam` | a camera |
| **Gradio app** | `make app` | — |
| **REST service** | `make serve` | — |
| **Notebook** | `demos/notebooks/blink_detection_rn30.ipynb` | the RN30 corpus |

**Start with `make video-demo`.** It needs no camera and no permissions, and its
output is checkable: the bundled clip has five annotated blinks (frames 170,
171, 227, 275, 276), so the plots should spike there.

## The layout

```
demos/
  video.py       offline runner -- the frame, plus a scrolling plot per eye
  webcam.py      the same view, live, on three threads
  gradio/        the Gradio app, also deployed as a Hugging Face Space
  docker/        the FastAPI service and its two Dockerfiles
  notebooks/     all four models on one RN30 window
```

`video.py` and `webcam.py` share their rendering through
`blinklinmult.live`, so the two views cannot drift apart.

## Related directories

`demos/` holds what you *run*. Two siblings hold what produced the numbers:

* **`experiments/`** — the arms, evaluations and report generators behind the
  published results. These read from `results/` and print tables.
* **`tools/`** — everything that *builds* an artifact: the ONNX exports, the
  scoring and comparison drivers, the benchmarks, the embedding cache.

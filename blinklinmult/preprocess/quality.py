"""Per-frame quality signals, and one score for ranking windows by them.

Nothing here filters anything. The builder's own gates decide what is written;
these values record *how marginal* a surviving crop was, so bad data can be
found and looked at instead of merely assumed absent.

**Why looking matters.** Thirty MPEblink test clips (``test/183`` onward) are
two-person profile conversations where the eye landmarks track correctly but
the eye itself is edge-on and invisible. Every automated check passed -- crop
contrast was *higher* than in good clips, and the median eye span, 21.5 px, is
perfectly ordinary. Only rendering the patches revealed it. A sortable score
turns that hour into a scroll.

**Signals are stored separately, never pre-combined.** How to weigh blur against
foreshortening against a bad contour fit is an empirical question, and the
answer is visible only once the corpora are built and the distributions can be
looked at. Baking a single number in at build time would freeze that choice into
65 GB of data; storing the components lets the dataloader decide, and lets the
decision change without re-preprocessing.

The signals are deliberately cheap and independent:

* :data:`~blinklinmult.data.schema.EYE_ON_SCREEN` -- has the eye left the frame?
* :data:`~blinklinmult.data.schema.EYE_SPAN` -- is there real detail, or is a
  64 px patch mostly interpolation?
* :data:`~blinklinmult.data.schema.EYE_ASPECT` -- is the face in profile? Span
  cannot see this; the contour flattening can.
* :data:`~blinklinmult.data.schema.BOX_SHIFTED` -- was the crop slid off centre
  to stay inside the frame?
* :func:`blur_score` -- is the lid edge destroyed by motion blur? The geometry
  cannot see this: the landmarks still fit a blurred eye perfectly.
* :func:`exposure_score` -- is the patch near-black or blown out?
* :func:`contour_fit` -- does the fitted contour sit on a real lid edge, or on
  flat skin? This is the one signal that catches the profile failure above.
* :func:`box_jitter` -- is the detector holding lock, or is the box skittering?

**Every signal describes one eye alone.** A sample is one eye, because the model
predicts eye-wise so that winks and per-eye patterns are detectable at all. A
signal read from the *other* eye would leak across samples, would be undefined
exactly when one eye is occluded, and -- for asymmetry especially -- would
penalise the very difference a wink consists of.
"""

from __future__ import annotations

import numpy as np

DEGENERATE_SPAN = 1e-6
"""Eye width below which a span carries no usable scale.

Not merely a division guard. Dividing a real displacement by this floor yields
a ratio in the hundreds of millions -- measured at 494 959 584 on RN30 -- which
silently violated the ``[0, 1]`` contract :data:`~blinklinmult.data.schema.QUALITY_SIGNALS`
documents. A span this small means the detector reported nothing, so the frame
is scored maximally unreliable instead.
"""

GOOD_EYE_SPAN = 24.0
"""Eye span, in pixels, at and above which resolution stops being a concern.

A 64 px crop covers :data:`~blinklinmult.preprocess.geometry.EYE_CROP_SCALE`
times the eye's width, so a 24 px eye already fills ~48 px of the patch and
upsampling adds little. Below it the patch is increasingly interpolation.
Measured medians run 21-41 px across MPEblink clips, so this sits deliberately
mid-range rather than at an extreme.
"""

FRONTAL_ASPECT_MAX = 0.55
"""Height/width ratio above which an eye is probably seen at an angle.

**Measured, and the opposite of the intuition.** A face turning to profile
compresses the eye contour *horizontally* while its height persists, so the
ratio **rises**. Over 97 000 MPEblink crops:

===========  ======  ======  ======
group        p10     p50     p90
===========  ======  ======  ======
frontal      0.222   0.364   0.538
profile      0.375   0.667   1.333
===========  ======  ======  ======

At this threshold 71% of profile frames are caught against 8.7% of frontal
ones. The overlap is real and the signal is a hint, never a verdict -- which is
why it only ranks windows for inspection.
"""

FRONTAL_ASPECT_MIN = 0.15
"""Ratio below which an eye is a flat slit.

The other tail: a closing lid drives the ratio towards zero, and so does a
badly-fitted contour. Legitimate mid-blink, which is again why this ranks
rather than rejects -- a low-aspect window may be exactly the blink the corpus
annotates.
"""

ASPECT_WEIGHT = 0.25
"""How much the profile signal pulls a window's confidence down.

Deliberately the smallest weight of the three, because a low aspect is the most
ambiguous signal here -- a genuine blink produces one too. Enough to sort
profile clips towards the bottom, not enough to bury a blinking frontal face.
"""

SPAN_WEIGHT = 0.25
"""How much low resolution pulls a window's confidence down."""

ON_SCREEN_WEIGHT = 0.5
"""How much leaving the frame pulls a window's confidence down.

The largest weight: an eye off the frame is the one failure here that is
unambiguous. The other two are matters of degree.
"""


def eye_aspect(points: np.ndarray) -> float:
    """Height-to-width ratio of an eye's landmark contour.

    Args:
        points (np.ndarray): ``(N, 2)`` eyelid contour points for one eye.

    Returns:
        float: Height divided by width, ``0.0`` when the contour is degenerate.
        Falls towards zero both as a face turns to profile and as the lid
        closes.
    """
    if points.size == 0:
        return 0.0
    lower, upper = points.min(axis=0), points.max(axis=0)
    width = float(upper[0] - lower[0])
    if width <= 0:
        return 0.0
    return float(upper[1] - lower[1]) / width


def on_screen_fraction(points: np.ndarray, height: int, width: int) -> float:
    """Fraction of an eye's landmarks lying inside the frame.

    Args:
        points (np.ndarray): ``(N, 2)`` eyelid contour points for one eye.
        height (int): Frame height in pixels.
        width (int): Frame width in pixels.

    Returns:
        float: ``1.0`` when every point is inside, ``0.0`` when none is.
    """
    if points.size == 0:
        return 0.0
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    return float(inside.mean())


def window_confidence(
    on_screen: np.ndarray,
    span: np.ndarray,
    aspect: np.ndarray,
    valid: np.ndarray | None = None,
) -> float:
    """Pool per-frame signals into one score for ranking windows.

    A weighted mean of three normalised terms, each clipped to ``[0, 1]`` so no
    single excellent frame can offset a bad one. Invalid frames are excluded
    rather than scored as zero: a window is judged on the frames it actually
    has, and its *proportion* of valid frames is already visible in the mask.

    **This is a ranking aid, not a probability and not a training weight.** It
    has no calibration; the only claim is that a lower score is worth looking at
    sooner.

    Args:
        on_screen (np.ndarray): ``(T,)`` fraction of eye landmarks in frame.
        span (np.ndarray): ``(T,)`` eye width in pixels.
        aspect (np.ndarray): ``(T,)`` eye height/width ratio.
        valid (np.ndarray | None): ``(T,)`` bool; ``None`` treats all as valid.

    Returns:
        float: Score in ``[0, 1]``; ``0.0`` when no frame is valid.
    """
    on_screen = np.asarray(on_screen, dtype=np.float64)
    span = np.asarray(span, dtype=np.float64)
    aspect = np.asarray(aspect, dtype=np.float64)

    keep = np.ones(on_screen.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if not keep.any():
        return 0.0

    screen_term = np.clip(on_screen[keep], 0.0, 1.0)
    span_term = np.clip(span[keep] / GOOD_EYE_SPAN, 0.0, 1.0)
    # Penalise departure from the frontal band in *either* direction: a ratio
    # too high means a face seen at an angle, too low means a flat slit. Inside
    # the band the term is 1.0 and contributes nothing.
    values = aspect[keep]
    above = np.clip((values - FRONTAL_ASPECT_MAX) / FRONTAL_ASPECT_MAX, 0.0, 1.0)
    below = np.clip((FRONTAL_ASPECT_MIN - values) / FRONTAL_ASPECT_MIN, 0.0, 1.0)
    aspect_term = 1.0 - np.maximum(above, below)

    score = (
        ON_SCREEN_WEIGHT * screen_term.mean()
        + SPAN_WEIGHT * span_term.mean()
        + ASPECT_WEIGHT * aspect_term.mean()
    )
    return float(np.clip(score, 0.0, 1.0))


SUSPECT_CONFIDENCE = 0.8
"""Score below which a window is worth inspecting by eye.

Not a rejection threshold -- nothing is dropped by it.
:data:`~blinklinmult.preprocess.geometry.MIN_EYE_ON_SCREEN` remains the only
hard gate in the pipeline.

Calibrated on 4 667 real MPEblink windows rather than chosen: it flags 36.6% of
windows from the known profile clips against 2.2% from known frontal ones, a
17x enrichment. That is what makes a worst-first ordering worth scrolling. It
does **not** flag most bad windows -- the distributions overlap heavily -- so
treat a high score as "not obviously broken", never as "verified".
"""


def is_suspect(confidence: float) -> bool:
    """Whether a window is worth a human look.

    Args:
        confidence (float): A :func:`window_confidence` score.

    Returns:
        bool: ``True`` below :data:`SUSPECT_CONFIDENCE`.
    """
    return confidence < SUSPECT_CONFIDENCE


GOOD_BLUR_RATIO = 0.05
"""Laplacian-to-intensity variance ratio at which sharpness stops being a concern.

Blink detection reads one thing: whether the eyelid edge is where the open eye's
sclera should be. Motion blur destroys exactly that edge while leaving every
geometric signal intact -- the landmarks still fit, the span is still 30 px, the
eye is still on screen -- so this is information no other signal here has.

Expressed as a **ratio** against the crop's own intensity variance rather than an
absolute Laplacian variance, because the absolute value scales with the pixel
range and with contrast: a hard edge measures 0.067 on a ``[0, 1]`` crop and
about 4300 on the same crop in ``[0, 255]``. The ratio is invariant to both, so
one constant serves whatever the builder happens to feed it.

Calibrated on synthetic edges: a hard edge scores well above this, a
whole-patch ramp scores 0.
"""

GOOD_EXPOSURE_RANGE = 0.25
"""Fraction of a crop's full range its values must span to be well exposed.

Catches the near-black and blown-out patches that occur when a face passes
through shadow or a light source: such a crop has no lid edge to read whatever
its landmarks claim.

Measured as ``(max - min)`` over the crop's own range rather than an absolute
standard deviation, for the same scale-invariance reason as
:data:`GOOD_BLUR_RATIO` -- the builder may store crops in ``[0, 1]`` or
``[0, 255]`` and one constant should serve both.
"""


def blur_score(crop: np.ndarray) -> float:
    """How sharp a crop is, in ``[0, 1]``.

    The variance of the Laplacian: a standard, cheap sharpness proxy. A blurred
    patch has little high-frequency content, so the second derivative is small
    everywhere and its variance collapses.

    Args:
        crop (np.ndarray): One eye crop, ``(C, H, W)`` or ``(H, W)``, any range.

    Returns:
        float: ``0`` for a featureless or motion-blurred patch, rising to ``1``
        at :data:`GOOD_BLUR_RATIO`. Zero for a degenerate crop, so a missing
        patch never scores as sharp.
    """
    values = np.asarray(crop, dtype=np.float64)
    if values.ndim == 3:
        values = values.mean(axis=0)
    if values.ndim != 2 or values.size < 9:
        return 0.0

    # 4-neighbour Laplacian by finite differences, which needs no convolution
    # dependency and is exact on the interior.
    interior = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1] + values[2:, 1:-1] + values[1:-1, :-2] + values[1:-1, 2:] - 4.0 * interior
    )
    # Normalised by the crop's own intensity variance, so the score means the
    # same thing whatever range the builder stores crops in.
    spread = float(values.var())
    if spread <= 0.0:
        return 0.0
    ratio = float(laplacian.var()) / spread
    return float(min(1.0, ratio / GOOD_BLUR_RATIO))


def exposure_score(crop: np.ndarray) -> float:
    """Whether a crop has usable dynamic range, in ``[0, 1]``.

    Args:
        crop (np.ndarray): One eye crop, ``(C, H, W)`` or ``(H, W)``.

    Returns:
        float: ``0`` when the patch is flat -- near-black, blown out, or
        otherwise featureless -- rising to ``1`` at :data:`GOOD_EXPOSURE_RANGE`.
    """
    values = np.asarray(crop, dtype=np.float64)
    if values.size == 0:
        return 0.0
    low, high = float(values.min()), float(values.max())
    if high <= low:
        return 0.0
    # Against the crop's own scale: a [0,255] patch and the same patch in [0,1]
    # must score identically.
    scale = max(abs(high), abs(low), 1e-6)
    return float(min(1.0, ((high - low) / scale) / GOOD_EXPOSURE_RANGE))


def contour_fit(crop: np.ndarray, contour: np.ndarray) -> float:
    """How well the fitted eyelid contour lands on real image structure.

    **The signal the others cannot provide.** Thirty MPEblink clips track eye
    landmarks "correctly" onto a face turned edge-on, where the eye itself is
    invisible: the span was ordinary, the crop contrast was *higher* than in good
    clips, and every automated check passed (see this module's header). What
    fails there is that the contour sits on skin rather than on a lid edge.

    Measured as the edge energy along the contour relative to the crop's overall
    edge energy. A contour tracing a genuine eyelid sits on a strong gradient; a
    hallucinated one sits on flat skin and scores near the crop average.

    Args:
        crop (np.ndarray): The eye crop, ``(C, H, W)`` or ``(H, W)``.
        contour (np.ndarray): ``(N, 2)`` eyelid points **in crop pixel
            coordinates**, not frame coordinates.

    Returns:
        float: ``0`` when the contour lies on featureless image, rising above
        ``1`` when it sits on structure stronger than the crop average. Clipped
        to ``[0, 1]``. Zero when the contour falls outside the crop entirely,
        which is itself a fit failure.
    """
    values = np.asarray(crop, dtype=np.float64)
    if values.ndim == 3:
        values = values.mean(axis=0)
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if values.ndim != 2 or values.size < 9 or points.size == 0:
        return 0.0

    gradient_y, gradient_x = np.gradient(values)
    magnitude = np.hypot(gradient_x, gradient_y)
    average = float(magnitude.mean())
    if average <= 0.0:
        return 0.0

    height, width = values.shape
    rows = np.clip(np.rint(points[:, 1]).astype(int), 0, height - 1)
    columns = np.clip(np.rint(points[:, 0]).astype(int), 0, width - 1)
    inside = (
        (points[:, 0] >= 0) & (points[:, 0] < width) & (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    if not inside.any():
        return 0.0

    along = float(magnitude[rows[inside], columns[inside]].mean())
    return float(min(1.0, along / average))


def box_jitter(centres: np.ndarray, spans: np.ndarray) -> np.ndarray:
    """Per-frame movement of the eye box, relative to the eye's own size.

    A detector losing lock produces a box that skitters between frames, and the
    resulting crops are misaligned even when each one looks fine in isolation.
    Normalised by eye span so a distant face is not penalised for the same
    pixel displacement that is negligible on a close one.

    Head motion moves the box legitimately, so a **high value is a hint, not a
    verdict** -- which is why this is stored as its own signal rather than folded
    into a single score at build time.

    **Saturating rather than unbounded.** One eye-width of movement between
    consecutive frames already means the crop no longer overlaps its
    predecessor; beyond that the signal cannot say anything more, and every
    larger value describes the same "lost lock" state. Reporting the raw ratio
    instead broke the ``[0, 1]`` contract :data:`QUALITY_SIGNALS` documents: a
    degenerate span divided by the ``1e-6`` floor produced **494 959 584** on
    RN30, which would dominate any loss weighting or normalisation it reached.

    A span at or below that floor is a detector failure, not a measurement, so
    it scores 1.0 -- maximally unreliable -- rather than manufacturing a number
    from a division by nothing.

    Args:
        centres (np.ndarray): ``(T, 2)`` box centres in frame pixels.
        spans (np.ndarray): ``(T,)`` eye widths in pixels.

    Returns:
        np.ndarray: ``(T,)`` displacement in eye-widths, clipped to ``[0, 1]``.
        The first frame has no predecessor and is scored ``0``.
    """
    points = np.asarray(centres, dtype=np.float64).reshape(-1, 2)
    widths = np.asarray(spans, dtype=np.float64).reshape(-1)
    steps = min(points.shape[0], widths.size)
    if steps < 2:
        return np.zeros(max(steps, 0), dtype=np.float64)

    moved = np.zeros(steps, dtype=np.float64)
    deltas = np.linalg.norm(np.diff(points[:steps], axis=0), axis=-1)
    # Against the *previous* frame's span, which is the scale the movement
    # happened at.
    previous = widths[: steps - 1]
    scale = np.maximum(previous, DEGENERATE_SPAN)
    ratio = deltas / scale
    # A span that never exceeded the floor carries no usable scale, so the
    # ratio derived from it is meaningless rather than merely large.
    moved[1:] = np.where(previous <= DEGENERATE_SPAN, 1.0, np.clip(ratio, 0.0, 1.0))
    return moved

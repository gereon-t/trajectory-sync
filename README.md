# sensor-sync

`sensor-sync` is a Python module for estimating the relative time offset between two position-producing sensors.

It is designed for streams where both sensors observe the same motion but do not agree perfectly on timestamp alignment. The module combines timestamp calibration, path-length tracking, interpolation, and a Kalman filter to estimate a stable temporal shift.

## What it provides

The public surface is centered around two main types:

- `Position`: one timestamped 3D sample with a scalar velocity
- `Synchronizer`: consumes primary and secondary sensor samples and maintains the current estimate

The Kalman state is maintained internally by `Synchronizer`.

The estimated state contains:

- `delta_t`: estimated time shift between the sensor streams
- `bias`: estimated path-length bias
- `sigma_delta_t`: uncertainty of the time-shift estimate
- `sigma_bias`: uncertainty of the bias estimate

## Estimation model

The synchronizer works in two phases.

During calibration, it computes a rough per-sensor wall-clock offset from incoming timestamps. After calibration, it shifts each sensor stream by its average offset and begins trajectory-based matching.

For each incoming sample, the module accumulates traveled path length. Secondary samples are matched against neighboring primary samples, and the primary trajectory is linearly interpolated to the secondary timestamp. The difference in path length becomes the measurement used by the Kalman filter.

The filter estimates two quantities:

- a time offset term
- a bias term that absorbs residual path-length drift

Conceptually, the measurement model is:

$$
\Delta s \approx v \cdot \Delta t + b
$$

where $\Delta s$ is the observed path-length difference, $v$ is interpolated velocity, $\Delta t$ is the time shift, and $b$ is a slowly varying bias.

## Installation

Requirements:

- Python 3.9 or newer

Using `uv`:

```powershell
uv sync
```

To activate the virtual environment on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

The project depends on:

- `numpy`
- `matplotlib`
- `trajectopy`

## Usage

Create a synchronizer and feed it timestamped position samples from both streams.

```python
from sensor_sync import Position, Synchronizer

synchronizer = Synchronizer(calibration_samples=20)

primary_sample = Position(
	timestamp=1000.0,
	x=0.0,
	y=0.0,
	z=0.0,
	v=2.5,
)

secondary_sample = Position(
	timestamp=1000.1,
	x=0.2,
	y=0.0,
	z=0.0,
	v=2.4,
)

synchronizer.on_new_position_sensor_primary(primary_sample)
synchronizer.on_new_position_sensor_secondary(secondary_sample)

state = synchronizer.state
print(state["delta_t"], state["bias"])
```

In a real integration, keep feeding both sensors as samples arrive. Once calibration has completed and enough primary samples are available for interpolation, the estimate will be updated automatically.

## Integration notes

- The synchronizer is thread-safe at the input boundary and guards updates with a lock.
- Incoming samples are expected to include velocity magnitude in `Position.v`.
- Matching assumes that the primary stream can provide surrounding samples for interpolation.
- Calibration currently depends on `time.time()` and incoming sample timestamps being comparable in the same clock domain.
- The implementation mutates `position.timestamp` in place after applying the sensor-specific average offset.

## Tuning

`Synchronizer` exposes three filter-tuning parameters at construction time:

- `delta_t_var`: process noise for time-shift stability
- `bias_var`: process noise for bias drift
- `delta_path_length_var`: measurement noise for path-length observations

These can be passed directly to the constructor alongside `calibration_samples` and `initial_dt`.

```python
synchronizer = Synchronizer(
	calibration_samples=20,
	initial_dt=0.0,
	delta_t_var=1e-12,
	bias_var=1e-3,
	delta_path_length_var=0.1,
)
```

Smaller process noise makes the estimate more stable but less responsive. Larger measurement noise makes the filter trust its prior more than new path-length observations.

## Module layout

- `sensor_sync.py`: core implementation of the estimator and synchronizer
- `pyproject.toml`: package metadata and dependencies

## Status

This repository currently exposes a single-module implementation intended for experimentation and integration work. It is not yet packaged as a multi-file library with a formal test suite or stable public API guarantees.

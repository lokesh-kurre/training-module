# Logging and Progress Tracking

This document explains the logging architecture and best practices for `training-module`, with emphasis on the vibe-coder principle: **minimize side effects and keep console output separate from file logging**.

## Architecture

The logging system follows a **dual-stream approach**:

1. **Console logging**: Rich output with progress bars, heartbeats, real-time metrics
2. **File logging**: Clean, machine-readable logs without progress bar clutter

## Progress Bars: Console Only

Progress bars are explicitly disabled from file output to keep logs clean and parseable.

### Configuration

In your config or CLI, control progress bar behavior:

```yaml
# cfg/default.yaml or experiments
trainer:
  show_progress_bar: true    # Display progress bar on console (stdout)
  timing_enabled: true       # Display timing breakdown (profile_timing kept as compatibility alias)
  log_batch_metrics: false   # Log per-batch metrics to file
  heartbeat_minutes: 15.0    # Log heartbeat every N minutes
```

### Console Output Example

```
Epoch 1 [########################--------] 24/40 loss=0.5432 acc=0.7891 lr=0.000100
Heartbeat: epoch=1, step=24/40, running_loss=0.5432, running_acc=0.7891, lr=0.000100
```

### File Output (log.txt)

```
[2026-05-14 10:30:45] [INFO] Heartbeat: epoch=1, step=24/40, running_loss=0.5432, running_acc=0.7891, lr=0.000100
[2026-05-14 10:45:00] [DEBUG] Batch 200/500: loss=0.4892
[2026-05-14 11:00:15] [INFO] Timing Breakdown (train epoch):
[2026-05-14 11:00:15] [INFO] timing.forward_loss: total=45.123s pct=35.2% avg_batch_ms=90.25
```

Notice: No progress bar characters (`[#---]`) in file output.

## Logging Levels

The framework supports configurable log levels:

```yaml
trainer:
  log_level: info  # Options: debug, info, warning, error
```

Levels filter what gets logged:

| Level | Console | File | Description |
|-------|---------|------|-------------|
| `debug` | All | All | Verbose: batch metrics, detailed timing |
| `info` | Core | Core | Default: epochs, heartbeats, summaries |
| `warning` | Warnings | Warnings | Only warnings (missing keys, etc.) |
| `error` | Errors | Errors | Only errors |

### Example: Debug Mode

```bash
python train.py --task classification --set trainer.log_level=debug
```

Output includes:
- Per-batch metrics
- Detailed timing breakdown
- Model summary statistics

### Example: Info Mode (Default)

```bash
python train.py --task classification --set trainer.log_level=info
```

Output includes:
- Epoch start/end
- Validation results
- Checkpointing summaries
- Heartbeat messages

## Run Directory Structure

Each run stores logs in a hierarchical structure:

```
run/<run_name>/
├── config_resolved.yaml      # Merged configuration
├── log.txt                   # Main log file (info + debug)
├── error.log                 # Warning/error/exception one-line entries
├── checkpoints/
│   ├── best.pt              # Best checkpoint
│   └── last.pt              # Last checkpoint
└── metrics/
    └── metrics.json         # Aggregated metrics
```

### Reading Logs

To tail progress in real-time:

```bash
# Real-time log viewing (updates every 1s)
tail -f run/<run_name>/log.txt

# View last 50 lines
tail -50 run/<run_name>/log.txt

# Search for warnings
grep "\[WARNING\]" run/<run_name>/log.txt

# Search for specific metric
grep "running_loss" run/<run_name>/log.txt
```

## Best Practices

### 1. Disable Progress Bars for Automated Tests

```yaml
# tests/configs/test_classification.yaml
trainer:
  show_progress_bar: false      # No console clutter
  timing_enabled: false         # Skip timing overhead
  log_batch_metrics: false      # No per-batch spam
```

### 2. Disable Heartbeats for Fast Runs

```yaml
# Quick experiments (1-2 minutes)
trainer:
  log_heartbeat: false          # Skip heartbeat overhead
```

Or enable for long runs (hours/days):

```yaml
trainer:
  log_heartbeat: true
  heartbeat_minutes: 30.0       # Log every 30 minutes
```

### 3. Use Logging for Debugging, Not Metrics

Metrics should go to **callbacks** (tensorboard, MLflow, WandB), not file logs.

Logs are for:
- Timestamps and progression
- Warnings (missing keys, device mismatches)
- Training state (epoch, step, learning rate)
- Errors and exceptions

Metrics belong in:
- TensorBoard event files
- MLflow runs
- WandB dashboard
- JSON metrics file

### 4. Log Important Config at Start

```python
# Trainer automatically logs:
# - Device (CPU/GPU)
# - Distributed rank (if DDP)
# - Resolved config path
# - Run directory
```

Check `run/<run_name>/config_resolved.yaml` for full merged config.

### 5. Avoid Blocking Console Output

Progress bars update in-place on console to avoid log spam:

```python
# Correct: progress bar overwrites itself
print(f"\rEpoch 1 [###-----] 30/100", end="", flush=True)

# Wrong: creates new line (clutters logs)
print(f"Epoch 1 [###-----] 30/100")
```

The framework uses in-place updates for progress bars.

## Trainer Logging Functions

### `_append_log(run_dir: Path, level: str, message: str, *, to_console: bool = False)`

Append a message to trainer logging sinks.

**Parameters**:
- `run_dir`: Run output directory
- `level`: Level string (`debug`, `info`, `warning`, `error`, ...)
- `message`: Message text
- `to_console`: If `True`, prints to terminal on rank-0

**Important behavior**:
- Progress-bar style lines are terminal-only and explicitly filtered out from file logs.
- `log.txt` receives concise debug/info lines.
- `error.log` receives warning/error/exception through logger handlers.

**Example**:
```python
self._append_log(run_dir, "info", f"Epoch {epoch} finished with loss={loss:.4f}")
self._append_log(run_dir, "debug", f"Batch {step}: timing={elapsed:.2f}s", to_console=True)
```

### `_format_progress_bar(current, total, width=28)`

Format a progress bar string (console-only).

**Returns**: `[###-----]` style string

**Example**:
```python
bar = self._format_progress_bar(30, 100)  # Returns "[##########----]"
print(f"\rEpoch 1 {bar} {30}/100", end="", flush=True)
```

### `_should_log(level)`

Check if a log level should be output based on config.

**Returns**: `True` if level >= configured log level

**Example**:
```python
if self._should_log("debug"):
    # Only executes if log_level is "debug"
    expensive_debug_info()
```

### `configure_run_error_log(run_dir: str | Path)`

Attach run-scoped one-line warning/error handler for all `training.*` loggers.

Source: [training/utils/logger.py](../training/utils/logger.py#L67)

## Callback Logging

Callbacks can also log to the file:

```python
class MyCallback(BaseCallback):
    def on_epoch_end(self, state):
        trainer = state.trainer  # Access trainer instance
        msg = f"Epoch {state.epoch}: val_loss={state.val_loss:.4f}"
        trainer._append_log(state.run_dir, "info", msg)
```

## Distributed Training Logging

In DDP mode, **only rank-0 logs** to file to avoid duplication:

```python
if self._is_rank0(runtime):
    print(message)  # Console (rank-0 only)
    self._append_log(run_dir, level, message)  # File (rank-0 only)
```

Other ranks silently discard logs.

## Performance Considerations

Progress bars and logging have overhead:

- **Progress bar rendering**: ~1ms per update
- **File I/O**: ~5-10ms per append (depends on disk)
- **Formatting**: ~0.1ms per message

For maximum performance in production:

```yaml
trainer:
  show_progress_bar: false      # Disable console rendering
  log_batch_metrics: false      # Skip per-batch file I/O
  timing_enabled: false         # Skip timing overhead
  log_heartbeat: false          # Skip periodic file writes
  log_level: info               # Filter out debug spam
```

Expected overhead reduction: **50-70% faster** training.

## Troubleshooting

### Log File Too Large

**Problem**: `log.txt` grows too large during long training.

**Solution**:
1. Reduce `log_batch_metrics: false` (default already)
2. Increase `heartbeat_minutes: 60.0` (log every hour instead)
3. Use log rotation in callbacks (custom callback):

```python
import logging
handler = logging.handlers.RotatingFileHandler(
    log_path, maxBytes=10_000_000, backupCount=3
)
```

### Missing Console Output

**Problem**: Progress bars don't appear.

**Causes**:
- `show_progress_bar: false` in config
- Running in non-TTY environment (redirected output)
- Rank != 0 in DDP (only rank-0 outputs)

**Check**:
```bash
# Enable progress bars
python train.py --set trainer.show_progress_bar=true

# Check rank in DDP
echo $LOCAL_RANK
```

### File Logs Polluted with Progress Bars

**Problem**: Log file contains `[###-----]` characters (should never happen).

**Root cause**: Bug in logging code (progress bars should never reach file).

**Report**: File an issue; should be filtered out by design.

## Related Configuration

See [PROJECT_STRUCTURE_AND_FLOW.md](./PROJECT_STRUCTURE_AND_FLOW.md) for config structure.

See [LIBRARY_USAGE.md](./LIBRARY_USAGE.md) for custom callback logging patterns.

See [Trainer class](../training/engine/trainer.py#L28) for implementation details.

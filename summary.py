import re
import sys
import torch
import types

def _color(text, color):
    colors = dict(
        cyan='\033[36m', green='\033[32m', yellow='\033[33m', red='\033[31m',
        blue='\033[34m', magenta='\033[35m', grey='\033[90m', white='\033[37m'
    )
    end = '\033[0m'
    return colors.get(color, '') + text + end if sys.stdout.isatty() else text

def strip_ansi(text):
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

class ExperimentSummary:
    def __init__(self, log_file=None):
        self.lines = []
        self.log_file = log_file

    def add_section(self, title):
        self.lines.append("")
        self.lines.append(_color("=" * 42, 'cyan'))
        self.lines.append(_color(f"{title:^42}", 'cyan'))
        self.lines.append(_color("=" * 42, 'cyan'))

    def add_kv(self, key, val, color='green'):
        keystr = f"{key:<18} | "
        valstr = f"{val}"
        self.lines.append(_color(keystr, color) + valstr)

    def add_raw(self, text, color=None):
        if color:
            text = _color(text, color)
        self.lines.append(text)

    def show(self):
        output = "\n".join(self.lines)
        print(output)
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(strip_ansi(output) + "\n")  # color stripped!

def summarize_training_data(train_df, cfg, log_file=None):
    summary = ExperimentSummary(log_file)
    summary.add_section("DATA SUMMARY")
    summary.add_kv("Samples", len(train_df))
    shape = tuple(cfg['data'].get('dataset_class_kwargs', {}).get('target_size', ['?']))
    summary.add_kv("Input shape", shape)
    if 'type' in train_df.columns:
        val_counts = dict(train_df['type'].value_counts())
        summary.add_kv("Type counts", val_counts, 'yellow')
    else:
        summary.add_kv("Type counts", "unknown", 'yellow')
    if 'quality' in train_df.columns:
        q = train_df['quality']
        summary.add_kv("Quality range", f"{q.min()} - {q.max()}", 'yellow')
    summary.show()

def print_module_summary(module, inputs, log_file=None, max_nesting=3, skip_redundant=True):
    """
    Print a StyleGAN3/torchinfo-style summary including: layer/module name, param/buffer counts, output shape, dtype.
    Writes the same output to log_file if not None.
    """
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks (non-recursive)
    entries = []
    nesting = [0]
    def pre_hook(_mod, _inputs):
        nesting[0] += 1
    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outs = [t for t in outs if isinstance(t, torch.Tensor)]
            entries.append(dict(mod=mod, outputs=outs))
    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module
    with torch.no_grad():
        outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers
    tensors_seen = set()
    for e in entries:
        e['unique_params'] = [t for t in e['mod'].parameters(recurse=False) if id(t) not in tensors_seen]
        e['unique_buffers'] = [t for t in e['mod'].buffers(recurse=False) if id(t) not in tensors_seen]
        e['unique_outputs'] = [t for t in e['outputs'] if id(t) not in tensors_seen]
        tensors_seen |= {id(t) for t in e['unique_params'] + e['unique_buffers'] + e['unique_outputs']}

    if skip_redundant:
        entries = [e for e in entries if len(e['unique_params']) or len(e['unique_buffers']) or len(e['unique_outputs'])]

    # Table header/rows setup
    rows = [["Module", "Parameters", "Buffers", "Output shape", "Dtype"]]
    rows += [['---'] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = '<top-level>' if e['mod'] is module else submodule_names[e['mod']]
        param_size = sum(t.numel() for t in e['unique_params'])
        buffer_size = sum(t.numel() for t in e['unique_buffers'])
        output_shapes = [str(list(t.shape)) for t in e['outputs']]
        output_dtypes = [str(t.dtype).split('.')[-1] for t in e['outputs']]
        rows += [[
            name + (':0' if len(e['outputs']) >= 2 else ''),
            str(param_size) if param_size else '-',
            str(buffer_size) if buffer_size else '-',
            (output_shapes + ['-'])[0],
            (output_dtypes + ['-'])[0],
        ]]
        for idx in range(1, len(e['outputs'])):
            rows += [[name + f':{idx}', '-', '-', output_shapes[idx], output_dtypes[idx]]]
        param_total += param_size
        buffer_total += buffer_size
    rows += [['---'] * len(rows[0])]
    rows += [['Total', str(param_total), str(buffer_total), '-', '-']]

    # Print table
    widths = [max(len(str(cell)) for cell in column) for column in zip(*rows)]
    output = "\n" + "\n".join(
        '  '.join(str(cell) + ' ' * (width - len(str(cell))) for cell, width in zip(row, widths)) for row in rows
    ) + "\n"
    print(output)
    if log_file:
        with open(log_file, "a") as f:
            f.write(output + "\n")
    return outputs

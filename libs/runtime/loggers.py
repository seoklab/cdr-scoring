# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: MIT

import pathlib
import time
import json
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Callable, Optional

import torch.distributed as dist
try:
    import dllogger
    from dllogger import Verbosity
except ModuleNotFoundError:
    dllogger = None
    Verbosity = None
try:
    import wandb
except ModuleNotFoundError:
    wandb = None

from runtime.utils import rank_zero_only


class Logger(ABC):
    @rank_zero_only
    @abstractmethod
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    @abstractmethod
    def log_metrics(self, metrics, step=None):
        pass

    @staticmethod
    def _sanitize_params(params):
        def _sanitize(val):
            if isinstance(val, Callable):
                try:
                    _val = val()
                    if isinstance(_val, Callable):
                        return val.__name__
                    return _val
                except Exception:
                    return getattr(val, "__name__", None)
            elif isinstance(val, pathlib.Path) or isinstance(val, Enum):
                return str(val)
            return val

        return {key: _sanitize(val) for key, val in params.items()}


class LoggerCollection(Logger):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def __getitem__(self, index):
        return [logger for logger in self.loggers][index]

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, step)

    @rank_zero_only
    def log_hyperparams(self, params):
        for logger in self.loggers:
            logger.log_hyperparams(params)

    @rank_zero_only
    def save_run_files(self, dataset_config=None, run_script=None):
        for logger in self.loggers:
            if hasattr(logger, 'save_run_files'):
                logger.save_run_files(dataset_config=dataset_config, run_script=run_script)


class DLLogger(Logger):
    def __init__(self, save_dir: pathlib.Path, filename: str):
        super().__init__()
        self.path = save_dir / filename
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            if dllogger is not None:
                dllogger.init(
                    backends=[dllogger.JSONStreamBackend(Verbosity.DEFAULT, str(self.path))])
            else:
                self.path.write_text("")

    @rank_zero_only
    def log_hyperparams(self, params):
        params = self._sanitize_params(params)
        self._log(step="PARAMETER", data=params)

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        if step is None:
            step = tuple()

        self._log(step=step, data=metrics)

    def _log(self, step, data):
        if dllogger is not None:
            dllogger.log(step=step, data=data)
            return
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps({"step": step, "data": data}, default=str) + "\n")


class WandbLogger(Logger):
    def __init__(
            self,
            name: str,
            save_dir: pathlib.Path,
            id: Optional[str] = None,
            project: Optional[str] = None,
            entity: Optional[str] = None,
            group: Optional[str] = None,
            job_type: Optional[str] = None,
            config: Optional[Dict[str, Any]] = None,
            require_online: bool = False,
    ):
        super().__init__()
        if wandb is None:
            raise ModuleNotFoundError("wandb is required when --wandb true")
        if not dist.is_initialized() or dist.get_rank() == 0:
            save_dir.mkdir(parents=True, exist_ok=True)
            # A wandb server error at init (2026-10-22: CommError "'NoneType' object has no
            # attribute 'get'" in send_run, wandb 0.16.6 vs a server without the ServerSettings
            # type) used to kill the job before the first step. Retry; then either abort
            # (require_online) or fall back to offline logging (sync later with `wandb sync`).
            kw = dict(name=name, project=project, id=id, dir=str(save_dir), resume='allow')
            for k, v in (('entity', entity), ('group', group), ('job_type', job_type), ('config', config)):
                if v is not None:
                    kw[k] = v
            self.experiment = None
            last_err = None
            for attempt in range(3):
                try:
                    self.experiment = wandb.init(**kw)
                    break
                except Exception as e:  # wandb raises CommError / UsageError / timeouts
                    last_err = e
                    print(f'[wandb] init failed (attempt {attempt + 1}/3): {type(e).__name__}: {e}', flush=True)
                    try:
                        wandb.finish()
                    except Exception:
                        pass
                    time.sleep(60)
            if self.experiment is None:
                if require_online:
                    raise RuntimeError(f'[wandb] online init failed 3x and --wandb_require_online is set: {last_err}')
                print('[wandb] falling back to offline mode (sync later with `wandb sync <run dir>`)', flush=True)
                self.experiment = wandb.init(mode='offline', **kw)
            _mode = getattr(getattr(self.experiment, 'settings', None), 'mode', 'online')
            if require_online and _mode != 'online':
                raise RuntimeError(f'[wandb] run is not online (mode={_mode}) and --wandb_require_online is set')
            print(f'[wandb] run {self.experiment.id} mode={_mode} url={getattr(self.experiment, "url", None)}',
                  flush=True)
            # separate x-axes: head-only prep and joint steps never collide with epochs
            try:
                r = self.experiment
                r.define_metric('epoch')
                r.define_metric('prep/step')
                r.define_metric('prep/*', step_metric='prep/step')
                r.define_metric('joint/step')
                r.define_metric('joint/*', step_metric='joint/step')
                for pat in ('train/*', 'valid/*', 'val/*', 'best/*', 'calib/*', 'loss/*'):
                    r.define_metric(pat, step_metric='epoch')
            except Exception as e:
                print(f'[wandb] define_metric failed: {e}', flush=True)

    @rank_zero_only
    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        params = self._sanitize_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        if step is not None:
            self.experiment.log({**metrics, 'epoch': step})
        else:
            self.experiment.log(metrics)

    @rank_zero_only
    def save_run_files(self, dataset_config: Optional[str] = None, run_script: Optional[str] = None) -> None:
        """Save dataset config YAML and run script to wandb for reproducibility.

        Files are copied into the wandb run directory so they appear in the
        Files tab. The YAML content is also stored as a dict in wandb.config
        so it's browsable directly in the Config tab.
        """
        import shutil, os
        run_dir = self.experiment.dir

        if dataset_config:
            cfg_path = os.path.abspath(dataset_config)
            if os.path.isfile(cfg_path):
                dst = os.path.join(run_dir, 'dataset_config.yaml')
                shutil.copy2(cfg_path, dst)
                wandb.save(os.path.join(run_dir, 'dataset_config.yaml'))
                try:
                    import yaml as _yaml
                    with open(cfg_path, encoding='utf-8') as f:
                        self.experiment.config.update(
                            {'dataset_yaml': _yaml.safe_load(f)}, allow_val_change=True
                        )
                except Exception:
                    pass

        if run_script:
            script_path = os.path.abspath(run_script)
            if os.path.isfile(script_path):
                basename = os.path.basename(script_path)
                dst = os.path.join(run_dir, basename)
                shutil.copy2(script_path, dst)
                wandb.save(os.path.join(run_dir, basename))
                with open(script_path, encoding='utf-8') as f:
                    self.experiment.config.update(
                        {'run_script_content': f.read()}, allow_val_change=True
                    )

        cmd = ' '.join(__import__('sys').argv)
        self.experiment.config.update({'full_command': cmd}, allow_val_change=True)

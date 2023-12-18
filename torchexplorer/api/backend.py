from __future__ import annotations

import os
import shutil
import json
import threading
import wandb
import sys

from torchexplorer.render import serialize
from torchexplorer.render.structs import NodeLayout


class Backend():
    def update(self, renderable: NodeLayout):
        raise NotImplementedError()


class WandbBackend(Backend):
    def __init__(self, watch_counter: int):
        self.watch_counter = watch_counter
        self.update_counter = 0
        self.delete_counter = 0
        self.delete_old_artifact_lag = 10

    def update(self, renderable: NodeLayout):
        if wandb.run is None:
            raise ValueError('Must call `wandb.init` before `torchexplorer.watch`')

        explorer_table, fields = self._wandb_table(renderable)

        chart = wandb.plot_table(
            vega_spec_name='spfrom_team/torchexplorer_v3d',
            data_table=explorer_table,
            fields=fields,
            string_fields={}
        )

        wandb.log({f'explorer_chart_{self.watch_counter}': chart}, commit=False)

        self._delete_old_artifacts()
        
        self.update_counter += 1

    def _wandb_table(
            self, renderable: NodeLayout
        ) -> tuple[wandb.Table, dict[str, str]]:

        rows = serialize.serialize_rows(renderable)
        fields = {key:key for key in rows[0]}
        keys = fields.keys() 
        data = [[row[key] for key in keys] for row in rows]
        table = wandb.Table(data=data, columns=list(keys))
        return table, fields
    
    def _delete_old_artifacts(self):
        lagged_version = self.update_counter - self.delete_old_artifact_lag
        delete_version = min(lagged_version, self.delete_counter)

        if delete_version >= 0:
            run_id = wandb.run.id
            artifact_name = f'run-{run_id}-explorer_chart_{self.watch_counter}_table'
            artifact_path = f'{artifact_name}:v{delete_version}'
            
            try:
                delete_artifact = wandb.run.use_artifact(artifact_path)
                delete_artifact.delete()
                self.delete_counter += 1
            except Exception as e:
                pass


class StandaloneBackend(Backend):
    def __init__(self, standalone_dir: str, standalone_port: int, verbose: bool):
        self.standalone_dir = standalone_dir
        self.standalone_port = standalone_port
        self.verbose = verbose

        source_explorer_dir = os.path.dirname(os.path.dirname(__file__))
        source_app_path = os.path.join(source_explorer_dir, 'standalone')
        target_app_path = os.path.abspath(standalone_dir)
        source_vega_path = os.path.join(source_explorer_dir, 'vega/vega_dataless.json')
        target_vega_path = os.path.join(target_app_path, 'vega_dataless.json')

        if os.path.exists(os.path.join(target_app_path, 'data', 'data.json')):
            os.remove(os.path.join(target_app_path, 'data', 'data.json'))

        if not os.path.exists(target_app_path):
            shutil.copytree(source_app_path, target_app_path)
            shutil.copyfile(source_vega_path, target_vega_path)

        if verbose:
            print(f'Starting TorchExplorer at http://localhost:{standalone_port}')

        # Launch flask app
        sys.path.insert(1, target_app_path)
        import app # type: ignore
        app.vega_spec_path = target_vega_path
        threading.Thread(target=lambda: app.app.run(port=standalone_port)).start()

    def update(self, renderable: NodeLayout):
        data = serialize.serialize_rows(renderable)
        target_app_path = os.path.abspath(self.standalone_dir)

        data_path = os.path.join(target_app_path, 'data', 'data.json')
        with open(data_path, 'w') as f:
            f.write(json.dumps(data))


class DummyBackend(Backend):
    def __init__(self):
        pass

    def update(self, renderable: NodeLayout):
        pass

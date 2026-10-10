"""Read NPZ headers only to distinguish generated shape errors from validator defects."""
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,'/tmp/auto_data_capability_project_20261010_v4')
import online_ts
import numpy as np
root=Path('/tmp/auto_data_capability_acceptance_20261010_v4/known_loop')
results=[]
for path in sorted((root/'processing').glob('*/outputs/processing.npz')):
    members=[]
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            with archive.open(info) as stream:
                version=np.lib.format.read_magic(stream)
                shape,order,dtype=np.lib.format.read_array_header_1_0(stream) if version==(1,0) else np.lib.format.read_array_header_2_0(stream)
            members.append({'name':info.filename,'shape':shape,'dtype':str(dtype),'expanded_bytes':info.file_size})
    results.append({'candidate_id':path.parent.parent.name,'members':members,'total_expanded_bytes':sum(row['expanded_bytes'] for row in members)})
print(json.dumps(results),flush=True)

"""Reuse the reviewed create-only publisher with an exact N1 destination binding."""
import ast
from pathlib import Path
import deploy_fourgroup_m1 as deployment


def main():
    stage=deployment.ROOT/'research_runs/agent_research/intervention_diagnosis/20261015_n1_path_engineering_r1'
    deployment.STAGE=stage
    deployment.build()
    old="pathlib.Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')"
    new="pathlib.Path('/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment')"
    for name in ('deployment_prepare.py','deployment_finish.py'):
        path=stage/'private'/name
        content=path.read_text(encoding='utf-8')
        assert content.count(old)==1
        content=content.replace(old,new)
        if name=='deployment_finish.py':
            before='fourgroup_runtime.configure();fourgroup_runtime.verify(root)'
            assert content.count(before)==1
            content=content.replace(before,'import n1_path_runtime\nfourgroup_runtime.configure();n1_path_runtime.verify(root)')
        ast.parse(content)
        path.write_text(content,encoding='utf-8')


if __name__=='__main__':
    main()

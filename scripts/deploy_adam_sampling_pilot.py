"""Bind the existing create-only publisher to this new experiment root."""
import ast
import deploy_fourgroup_m1 as deployment


def main():
    stage = deployment.ROOT/'research_runs/agent_research/intervention_diagnosis/20261015_adam_sampling_pilot_r1'
    deployment.STAGE = stage
    deployment.build()
    old = "pathlib.Path('/root/auto_data_experiments/20261015_fourgroup_m1_r1')"
    new = "pathlib.Path('/root/auto_data_experiments/20261015_adam_sampling_pilot_r1')"
    for name in ('deployment_prepare.py', 'deployment_finish.py'):
        path = stage/'private'/name
        content = path.read_text(encoding='utf-8')
        assert content.count(old) == 1
        content = content.replace(old, new)
        if name == 'deployment_finish.py':
            before = 'fourgroup_runtime.configure();fourgroup_runtime.verify(root)'
            assert content.count(before) == 1
            content = content.replace(before, 'import adam_sampling_pilot\nfourgroup_runtime.configure();adam_sampling_pilot.verify(root)')
        ast.parse(content)
        path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()

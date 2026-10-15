"""Bind the reviewed immutable archive builder to the separate N1 project."""
import argparse
from pathlib import Path
import archive_fourgroup_m1 as delivery

ROOT=Path('/root/auto_data_experiments/20261015_n1_path_engineering_r1/experiment')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    args=parser.parse_args()
    if not __debug__:
        raise RuntimeError('Unoptimized archive generation required')
    delivery.safe_path(ROOT)
    plan=delivery.read(ROOT/'plan.json')
    assert plan['stage']=='n1_path_engineering_r1' and not plan['scientific_children_enabled']
    phases=delivery.closed_phases(ROOT)
    assert phases and set(phases).issubset({'preflight','paths','audit_paths','fd','audit_fd'})
    failure=(ROOT/'stage_failure.json').is_file() or (ROOT/'fd_failure.json').is_file()
    if not failure:
        assert delivery.read(ROOT/'paths_audit.json')['status']=='PASS'
        if 'fd' in phases:
            assert 'audit_fd' in phases and delivery.read(ROOT/'fd_audit.json')['status']=='PASS'
    # Existing builder already verifies all source/input locks, closed controls,
    # ordinary files, immutable signatures, capacity and every tar member hash.
    delivery.ROOT=ROOT
    print(delivery.create_archive(ROOT,args.name))


if __name__=='__main__':
    main()

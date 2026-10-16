"""Build transparent ETT tables from all registered, audited result rows."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
STAGE=ROOT/'research_runs/ett_confirmation'
SEEDS=(2026101901,2026101902,2026101903)
CHANNELS=('HUFL','HULL','MUFL','MULL','LUFL','LULL','OT')


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')


def csv_table(path,rows):
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def station_report(station):
    directory=STAGE/station
    assert read(directory/'audit.json')['status']=='PASS'
    rows=read(directory/'result_rows.json');summary=read(directory/'summary.json')
    assert len(rows)==30
    expected={(seed,action,impl,label) for seed in SEEDS for action in ('PARENT','FAVOR_EARLY','FAVOR_LATE')
        for impl in (('P',) if action=='PARENT' else ('S','W')) for label in ('best','last')}
    assert {(r['seed'],r['action'],r['implementation'],r['checkpoint']) for r in rows}==expected
    gains=[];channels=[]
    for r in rows:
        parent=next(p for p in rows if p['seed']==r['seed'] and p['checkpoint']==r['checkpoint'] and p['action']=='PARENT')
        base={k:r[k] for k in ('seed','action','implementation','checkpoint','step')}
        gains.append(dict(station=station,**base,NMSE=r['NMSE'],NMAE=r['NMAE'],
            NMSE_gain_percent=100*(parent['NMSE']-r['NMSE'])/parent['NMSE'],
            NMAE_gain_percent=100*(parent['NMAE']-r['NMAE'])/parent['NMAE']))
        for c,name in enumerate(CHANNELS):
            channels.append(dict(station=station,**base,channel=name,NMSE=r['channel_NMSE'][c],NMAE=r['channel_NMAE'][c],
                NMSE_gain_percent=100*(parent['channel_NMSE'][c]-r['channel_NMSE'][c])/parent['channel_NMSE'][c],
                NMAE_gain_percent=100*(parent['channel_NMAE'][c]-r['channel_NMAE'][c])/parent['channel_NMAE'][c]))
    aggregate=[]
    for label in ('best','last'):
        result=summary['results'][label]
        for cell in result['cells']:
            selected=[r for r in gains if r['checkpoint']==label and r['action']==cell['action'] and r['implementation']==cell['implementation']]
            value=np.mean([r['NMSE_gain_percent'] for r in selected])
            np.testing.assert_allclose(value,cell['mean_gain_percent'],rtol=0,atol=1e-12)
            aggregate.append(dict(station=station,checkpoint=label,action=cell['action'],implementation=cell['implementation'],
                mean_NMSE=float(np.mean([r['NMSE'] for r in selected])),mean_NMAE=float(np.mean([r['NMAE'] for r in selected])),
                mean_NMSE_gain_percent=float(value),min_seed_gain_percent=min(r['NMSE_gain_percent'] for r in selected),
                max_seed_gain_percent=max(r['NMSE_gain_percent'] for r in selected),
                mean_NMAE_gain_percent=float(np.mean([r['NMAE_gain_percent'] for r in selected])),
                mean_response_category='beneficial' if value>.25 else 'harmful' if value<-.25 else 'unchanged',
                adoption_1pct=bool(value>=1.)))
        for pair in result['paired_differences']:
            s=next(c for c in result['cells'] if c['action']==pair['action'] and c['implementation']=='S')
            w=next(c for c in result['cells'] if c['action']==pair['action'] and c['implementation']=='W')
            delta=np.array(w['gains_percent'])-s['gains_percent']
            np.testing.assert_allclose(delta,pair['W_minus_S_pp'],rtol=0,atol=1e-12)
        # Recompute shared-prediction lower bound and all deterministic decisions.
        lower=np.mean([abs(p['mean_difference_pp'])/2 for p in result['paired_differences']])
        np.testing.assert_allclose(lower,result['shared_mean_prediction_MAE_lower_bound_pp'],rtol=0,atol=1e-12)
        for impl in ('S','W'):
            pool=[('PARENT',0.)]+[(c['action'],c['mean_gain_percent']) for c in result['cells'] if c['implementation']==impl]
            assert max(pool,key=lambda x:x[1])[0]==result['choices'][impl]
    csv_table(directory/'responses.csv',gains)
    csv_table(directory/'channels.csv',channels)
    csv_table(directory/'aggregate.csv',aggregate)
    preflight=read(directory/'preflight.json');training=read(directory/'training_complete.json');audit=read(directory/'audit.json')
    cost=dict(full_trainings=15,diagnostic_updates=preflight['diagnostic_updates'],full_trajectory_replays=0,
        preflight_measured_seconds=preflight['total_seconds'],training_and_C_seconds=training['total_train_and_C_wall_seconds'],
        audit_measured_seconds=audit['audit_seconds'])
    phases={}
    for phase in ('preflight','train','audit'):
        control=read(STAGE/'control'/f'{station}_{phase}'/'exit.json')
        assert control['exit_code']==0
        phases[phase]=control['finished_unix']-control['started_unix']
    cost['process_wall_seconds_by_phase']=phases
    cost['total_process_wall_seconds']=sum(phases.values())
    write(directory/'costs.json',cost)
    report=[f'# {station}完整确认结果','',
        '原冻结15项完整训练全部结束；本报告包括全部动作、实现和种子，不删除负结果。',
        f"累计完整训练为{449 if station=='ETTh1' else 464}；本阶段新增15。101个诊断更新另计，完整训练重放0。",'',
        '## 主结果：按B选best，在C评价','',
        '|动作|实现|平均NMSE改善%|三种子范围%|平均NMAE改善%|1%采用|',
        '|---|---|---:|---:|---:|---|']
    for r in aggregate:
        if r['checkpoint']=='best':report.append(f"|{r['action']}|{r['implementation']}|{r['mean_NMSE_gain_percent']:.6f}|[{r['min_seed_gain_percent']:.6f}, {r['max_seed_gain_percent']:.6f}]|{r['mean_NMAE_gain_percent']:.6f}|{r['adoption_1pct']}|")
    best=summary['results']['best']
    report+=['',f"S最优动作：{best['choices']['S']}；W最优动作：{best['choices']['W']}。",
        f"S/W分别的oracle收益：{best['separate_oracles_percent']['S']:.6f}% / {best['separate_oracles_percent']['W']:.6f}%。",
        f"固定近期优先策略遗憾：{best['fixed_late_regret_pp']['S']:.6f} / {best['fixed_late_regret_pp']['W']:.6f}个百分点。",
        f"联合oracle：{best['joint_oracle_percent']:.6f}%；它是已观察上界，不能作为可部署方法收益。",'',
        '## 两个分开的命题','',
        f"至少一个动作的三种子W−S同向超过0.25pp：{best['response_divergence_replicated']}。",
        f"最优动作改变：{best['choice_changed']}；至少一个动作的1%采用资格改变：{best['any_adoption_changed']}。"]
    for p in best['paired_differences']:
        report.append(f"{p['action']}的W−S（pp）："+', '.join(f'{v:.6f}' for v in p['W_minus_S_pp'])+f"；均值{p['mean_difference_pp']:.6f}。")
    report+=['',f"同μ共享预测的四格MAE下界：{best['shared_mean_prediction_MAE_lower_bound_pp']:.6f}pp。这是此有限池上的数值下界。",'',
        '## 成本、检查与边界','',
        f"三阶段进程总耗时{cost['total_process_wall_seconds']:.3f}秒，含解释器和初始化；训练及C评价内部计时{cost['training_and_C_seconds']:.3f}秒。发布、传输和人工实现耗时不包含在此运行计时中。",
        '来源、全部窗口、A归一化、p/w、真实损失和Adam一步检查通过；15个初始模型及best/last的B/C指标重算通过。',
        'responses.csv保留每种子原始指标及收益；channels.csv包含全部七通道，包括OT和退化通道；aggregate.csv与summary.json保留best/last两种口径。',
        '三种子同向是描述性稳定性，不称统计显著。两个站点同属ETT家族；结果不代表所有时间序列、模型或动作。D测量值未用于评价。',
        'ETTh1为开发复核站，ETTh2为预定确认站。不能根据last改选主检查点或根据ETTh1效果调整ETTh2。','']
    (directory/'REPORT.md').write_text('\n'.join(report),encoding='utf-8',newline='\n')
    write(directory/'table_check.json',dict(status='PASS',rows=len(rows),responses=len(gains),channel_rows=len(channels),
        summary_sha256=sha(directory/'summary.json'),rows_sha256=sha(directory/'result_rows.json'),
        scope='local arithmetic and registered coverage; no training replay'))
    return cost


def final_report():
    stations=('ETTh1','ETTh2')
    summaries={s:read(STAGE/s/'summary.json') for s in stations}
    costs={s:read(STAGE/s/'costs.json') for s in stations}
    for s in stations:
        assert read(STAGE/s/'audit.json')['status']=='PASS'
        assert read(STAGE/s/'table_check.json')['status']=='PASS'
    h2=summaries['ETTh2']['results']['best']
    verdict=dict(response_divergence_confirmed=h2['response_divergence_replicated'],
        optimal_action_changed=h2['choice_changed'],any_adoption_eligibility_changed=h2['any_adoption_changed'],
        confirmation_source='ETTh2',primary_checkpoint='B-selected best',
        new_full_trainings=30,cumulative_full_trainings=464,full_trajectory_replays=0,
        diagnostic_updates=sum(c['diagnostic_updates'] for c in costs.values()),
        total_process_wall_seconds=sum(c['total_process_wall_seconds'] for c in costs.values()),
        data_families_added=1,measurement_D_or_test_evaluations=0,
        limitation='two related stations, two predeclared actions, one backbone/horizon/optimizer; descriptive three-seed criterion')
    write(STAGE/'confirmation.json',verdict)
    write(STAGE/'final_costs.json',dict(**verdict,station_costs=costs,
        implementation_and_publication_transfer_time_seconds=None,
        unmeasured_time_note='not separately instrumented; excluded from measured process totals, not zero'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.2),layout='constrained')
    colors={'S':'#2463a8','W':'#ba5037'}
    for row,s in enumerate(stations):
        result=summaries[s]['results']['best']
        ax=axes[row,0]
        for j,impl in enumerate(('S','W')):
            for i,action in enumerate(('FAVOR_EARLY','FAVOR_LATE')):
                c=next(c for c in result['cells'] if c['action']==action and c['implementation']==impl)
                x=i+(j-.5)*.24
                ax.scatter(np.full(3,x),c['gains_percent'],s=23,alpha=.55,color=colors[impl])
                ax.plot([x-.08,x+.08],[c['mean_gain_percent']]*2,color=colors[impl],lw=3,
                    label=('Sampling (S)' if impl=='S' else 'Weighting (W)') if i==0 else None)
        ax.axhline(0,color='#666666',lw=.8)
        ax.axhline(1,color='#999999',ls='--',lw=.8,label='1% adoption')
        ax.set_xticks([0,1],['Favor early','Favor late']);ax.set_xlim(-.5,1.5)
        ax.set_ylabel('NMSE improvement (%)');ax.set_title(s+' | response relative to parent')
        ax.grid(axis='y',alpha=.15)
        if row==0:ax.legend(frameon=False,fontsize=8)
        ax=axes[row,1]
        for i,p in enumerate(result['paired_differences']):
            ax.scatter(np.full(3,i),p['W_minus_S_pp'],s=28,color='#39775b')
            ax.plot([i-.1,i+.1],[p['mean_difference_pp']]*2,lw=3,color='#194d37')
        ax.axhline(0,color='#666666',lw=.8);ax.axhspan(-.25,.25,color='#dddddd',alpha=.5)
        ax.set_xticks([0,1],['Favor early','Favor late']);ax.set_xlim(-.5,1.5)
        ax.set_ylabel('W minus S improvement (pp)');ax.set_title(s+' | matched-seed differences')
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('ETT confirmation: full predeclared response matrix\nDots: three seeds; horizontal marks: means; B-selected checkpoints',fontsize=13)
    fig.savefig(STAGE/'confirmation.png',dpi=160)
    fig.savefig(STAGE/'confirmation.pdf')
    plt.close(fig)
    lines=['# ETT确认完整报告','',
        '按原冻结设计完成ETTh1和ETTh2各15项完整训练，新增30，累计464。全部运行成功；没有按结果增加动作、倍率、种子或训练步数。',
        'ETTh1源码与完整结果发布核验后才解封ETTh2，解封依据见ETTh2_unlock.json。旧批次和已取消Adam pilot没有重启。','',
        '## 两个预定命题','',
        f"ETTh2主口径：响应分歧复现={h2['response_divergence_replicated']}；最优动作改变={h2['choice_changed']}；至少一项1%采用资格改变={h2['any_adoption_changed']}。",'',
        '|站点|动作|S平均改善%|W平均改善%|W−S均值pp|三种子同向超0.25pp|',
        '|---|---|---:|---:|---:|---|']
    for s in stations:
        result=summaries[s]['results']['best']
        for p in result['paired_differences']:
            values={impl:next(c['mean_gain_percent'] for c in result['cells'] if c['action']==p['action'] and c['implementation']==impl) for impl in ('S','W')}
            lines.append(f"|{s}|{p['action']}|{values['S']:.6f}|{values['W']:.6f}|{p['mean_difference_pp']:.6f}|{p['same_direction_beyond_quarter_pp']}|")
    lines+=['','## 决策与改善空间','',
        '|站点|S最优动作|W最优动作|S oracle %|W oracle %|联合oracle %|',
        '|---|---|---|---:|---:|---:|']
    for s in stations:
        r=summaries[s]['results']['best']
        lines.append(f"|{s}|{r['choices']['S']}|{r['choices']['W']}|{r['separate_oracles_percent']['S']:.6f}|{r['separate_oracles_percent']['W']:.6f}|{r['joint_oracle_percent']:.6f}|")
    lines+=['','oracle包含均匀父，三种子先按每个共同动作求平均再选择。oracle是观察到的上界，不是方法收益。固定近期策略的遗憾及联合池两种并列顺序见各站summary.json。',
        '主结果使用B选best，last及NMAE、七通道包括OT的所有退化均保留在各站表格，不能用次要指标覆盖主结果。','',
        '## 结论范围与研究位置','']
    if not h2['response_divergence_replicated']:
        lines.append('本次未在确认站复现预定强度的跨种子一致响应分歧。已有Hydraulic发现需保持局部适用范围；不能降低阈值或追加动作追求复现。')
    elif h2['choice_changed']:
        lines.append('确认站同时出现稳定响应分歧和最优动作改变，支持进一步研究实现方式如何影响数据决策；本轮没有测试新的代理或选择算法，不能据此宣称新方法胜出。')
    else:
        lines.append('确认站出现预定强度的响应分歧，但独立S/W池的最优动作没有改变，支持在这一有限池中区分响应辨识与动作选择。采用资格变化需独立解释。')
    lines+=['两个站点同属ETT家族，不算两个独立来源。只有两种动作、一个固定骨干、一个预测尺度和一种优化器；三种子判据是描述性稳定性，不是统计显著。结果没有证明方差迹是唯一原因，也不能概括所有数据干预。',
        '本轮用于确认分析命题，不承诺采样方法优于基线。后续首先结合旧Hydraulic证据收窄论文主张；不自动扩展候选、尺度、模型或种子。','',
        '## 成本与验收','',
        f"共30条完整训练、{verdict['diagnostic_updates']}次预检更新、0条完整训练重放；三阶段进程累计{verdict['total_process_wall_seconds']:.3f}秒。各站成本见costs.json，整体见final_costs.json。实现、发布和传输未单独计时，不记为零。",
        '两站必要数据、数值、完整覆盖和指标重算验收均通过。原始测量表和变换表不进入公开归档；公开检查点、采样计数、B曲线、指标、协议与源码。D/test未评价。','',
        '完整逐种子及通道结果见ETTh1/和ETTh2/；运行器源码发布收据、两站发布收据及Release资产收据共同确定同步状态。','']
    (STAGE/'FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    return verdict


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('station',choices=('ETTh1','ETTh2','all'))
    args=parser.parse_args();print(json.dumps(final_report() if args.station=='all' else station_report(args.station)))

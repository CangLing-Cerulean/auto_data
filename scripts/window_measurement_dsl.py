"""Bounded arithmetic interpreter for parent-only window measurements; never eval."""
import ast
from dataclasses import dataclass
import numpy as np
from parent_measurement_tables import INPUT_COLUMNS,ERROR_COLUMNS

@dataclass(frozen=True)
class Value:
    data: object
    axis: str='scalar'

def environment(table,features):
    result={name:Value(float(features[name])) for name in ('exposure_shift_first','removed_fraction','weighted_implementation')}
    for axis,prefix in [('A','A_inputs'),('B','B_inputs')]:
        for j,name in enumerate(INPUT_COLUMNS):result[axis+'_'+name]=Value(table[prefix][:,j],axis)
    errors=np.mean([table[k] for k in table if k.startswith('B_errors_s')],axis=0)
    for j,name in enumerate(ERROR_COLUMNS):result['E_'+name]=Value(errors[:,j],'B')
    result['E_contrast']=Value(errors[:,1]-errors[:,2],'B')
    return result

def evaluate(expression,table,features):
    env=environment(table,features);tree=ast.parse(expression,mode='eval')
    if len(list(ast.walk(tree)))>95:raise ValueError('Expression exceeds 95 AST nodes')
    def combine(a,b,op):
        if a.axis!=b.axis and 'scalar' not in (a.axis,b.axis):raise ValueError('Cross-window axes cannot be combined')
        return Value(op(a.data,b.data),b.axis if a.axis=='scalar' else a.axis)
    def visit(n):
        if isinstance(n,ast.Expression):return visit(n.body)
        if isinstance(n,ast.Name) and n.id in env:return env[n.id]
        if isinstance(n,ast.Constant) and type(n.value) in (int,float) and n.value in (-1,0,.5,1,2):return Value(float(n.value))
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):
            a=visit(n.operand);return Value(-a.data,a.axis)
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult)):
            return combine(visit(n.left),visit(n.right),{ast.Add:np.add,ast.Sub:np.subtract,ast.Mult:np.multiply}[type(n.op)])
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and not n.keywords:
            name=n.func.id;args=[visit(x) for x in n.args]
            if name=='safe_ratio' and len(args)==2:return combine(*args,lambda a,b:a/(1+abs(b)))
            if len(args)!=1:raise ValueError('Invalid function or arity')
            a=args[0]
            if name in ('abs','tanh','square','log1p_abs'):
                op={'abs':np.abs,'tanh':np.tanh,'square':np.square,'log1p_abs':lambda x:np.log1p(abs(x))}[name]
                return Value(op(a.data),a.axis)
            if name in ('mean','std','center','unit') and a.axis!='scalar':
                if name=='mean':return Value(float(np.mean(a.data)))
                if name=='std':return Value(float(np.std(a.data)))
                x=a.data-np.mean(a.data)
                return Value(x/(1+np.std(a.data)) if name=='unit' else x,a.axis)
            if name in ('first_mean','second_mean') and a.axis=='A':
                mask=table['A_first'] if name=='first_mean' else ~table['A_first']
                return Value(float(np.mean(a.data[mask])))
        raise ValueError('Unapproved expression')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        answer=visit(tree)
    if answer.axis!='scalar' or not np.isfinite(answer.data):raise ValueError('Measurement must be a finite scalar')
    return float(answer.data)

def summary(table):
    a=table['A_inputs'];b=table['B_inputs'];e=np.mean([table[k] for k in table if k.startswith('B_errors_s')],axis=0)
    e=np.column_stack([e,e[:,1]-e[:,2]]);names=(*ERROR_COLUMNS,'contrast');rows=[]
    for j,name in enumerate(INPUT_COLUMNS):
        cuts=np.quantile(b[:,j],[.25,.5,.75]);bins=np.searchsorted(cuts,b[:,j],side='right')
        sd=np.std(b[:,j]);ec=e-e.mean(axis=0)
        correlation=np.divide(np.mean((b[:,j]-b[:,j].mean())[:,None]*ec,axis=0),sd*np.std(e,axis=0),out=np.zeros(e.shape[1]),where=sd*np.std(e,axis=0)>0)
        rows.append(dict(input=name,A_first_mean=float(a[table['A_first'],j].mean()),A_second_mean=float(a[~table['A_first'],j].mean()),B_mean=float(b[:,j].mean()),B_std=float(sd),B_quartiles=cuts.tolist(),B_error_correlations=dict(zip(names,correlation.tolist())),B_quartile_error_means=[dict(count=int((bins==q).sum()),errors=None if not (bins==q).any() else dict(zip(names,e[bins==q].mean(axis=0).tolist()))) for q in range(4)]))
    return dict(A_windows=len(a),B_windows=len(b),statistics=rows)

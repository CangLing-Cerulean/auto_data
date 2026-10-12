import torch

def transform(x,state):
    y=x.clone()
    for d,s,sign in zip(state['dest'],state['source'],state['sign']):
        own_last=x[:,-1:,d]
        source_last=x[:,-1:,s]
        y[:,:,d]=own_last+state['own_gain']*(x[:,:,d]-own_last)+state['source_gain']*sign*(x[:,:,s]-source_last)
    return y

import numpy as np
import gym
import pdb

class Flatter(gym.spaces.Tuple):
    def __init__(self, space):
        self._singleton = not isinstance(space, (gym.spaces.Tuple, gym.spaces.Dict))
        self.addressed_spaces = flat_address_space(space)
        self.recipe = make_empty_object(space)
        super().__init__([subspace for subspace in self])

    def __iter__(self):
        for path, space in self.addressed_spaces:
            yield space
            
    def flatten(self, item):
        res = []
        for (path, space) in self.addressed_spaces:
            item_ = item
            for el in path:
                item_ = item_[el[1]]
            if isinstance(space, gym.spaces.Discrete):
                if isinstance(item_, str): # minerl's silly enum space
                    if item_ not in space.values and 'other' in space.values:
                        item_ = 'other'
                    res.append(np.eye(space.n)[space.values.index(item_)])
                else:
                    if not np.isscalar(item_) and item_.shape[-1] == space.n:
                        res.append(item_)
                    else:
                        res.append(np.eye(space.n)[item_])
            elif isinstance(space, gym.spaces.Box):
                res.append(item_)
            else:
                raise ValueError
        return res
    
    def unflatten(self, item):
        if self._singleton:
            return item[0]
        res = cook(self.recipe)
        for el, (key_path, space) in zip(item, self.addressed_spaces):
            ptr_ = res
            for k in key_path[:-1]:
                ptr_ = ptr_[k[1]]
            ptr_[key_path[-1][1]] = el
        return res

def flat_address_space(space, path=[]):
    if isinstance(space, gym.spaces.Tuple):
        return [kspace for k, subspace in enumerate(space.spaces)
                for kspace in flat_address_space(subspace, [*path, ('tuple', k)])]
    elif isinstance(space, gym.spaces.Dict):
        return [kspace
               for k, subspace in space.spaces.items()
               for kspace in flat_address_space(subspace, [*path, ('dict', k)])]
    else:
        return [(path, space)]

def make_empty_object(space, path=[]):
    if isinstance(space, gym.spaces.Tuple):
        return [(path, 'tuple', len(space.spaces)), 
                *[x for k, subspace in enumerate(space.spaces) if isinstance(subspace, (gym.spaces.Dict, gym.spaces.Tuple))
                    for x in make_empty_object(subspace, [*path, ('tuple', k)]) ]]
    elif isinstance(space, gym.spaces.Dict):
        return [(path, 'dict', list(space.spaces.keys())),
                *[x for k, subspace in space.spaces.items() if isinstance(subspace, (gym.spaces.Dict, gym.spaces.Tuple))
                    for x in make_empty_object(subspace, [*path, ('dict', k)]) ]]
    else:
        return None
    
def cook(recipe):
    if len(recipe)==0:
        return None
    (base_path, container_type, container_info) = recipe[0]
    if recipe[0][1] == 'dict':
        res = dict()
    elif recipe[0][1] == 'tuple':
        res = [None] * recipe[0][2]
    else:
        return None
    
    for (key_path, container_type, container_info) in recipe[1:]:
        ptr_ = res
        for _, k in key_path[:-1]:
            ptr_ = ptr_[k]
        if container_type == 'dict':
            ptr_[key_path[-1][1]] = dict()
        else:
            ptr_[key_path[-1][1]] = [None] * container_info
    return res


    
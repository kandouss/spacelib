import os, warnings
import numpy as np
import gym

class ArraySpec:
    def __init__(self, shape, dtype=np.float32, length=None):
        if isinstance(shape, tuple):
            self.shape = shape
        elif isinstance(shape, int):
            self.shape = (shape, )
        else:
            raise ValueError(f"ArraySpec shapes must be int or tuple (got {shape} ({type(shape)}))")
        self.length = length
        self.dtype = dtype
        self.n = int(np.prod(self.shape))
    
    @classmethod
    def from_array(cls, arr):
        ''' Create an arrayspec from an array.'''
        assert isinstance(arr, np.ndarray)
        return ArraySpec(arr.shape[1:], length=arr.shape[0], dtype=arr.dtype)
    
    def __repr__(self):
        return f"ArraySpec(shape={self.shape}, dtype={self.dtype}, length={self.length})"
    
    def coerce(self, buffer):
        if buffer.shape == self.shape:
            return buffer
        elif len(buffer.shape) == 2:
            return buffer.reshape(-1, *self.shape)
        else:
            return buffer.reshape(self.shape)

class ArrayContainer:
    def __init__(self, dtype, max_size=2048):
        self.dtype = dtype
        self.n = 0
        self.max_size = max_size
        self.specs = []
        self.spec_slices = []
        self.buffer = None
        self.length = None
        
    def fits(self, spec):
        if self.buffer is not None:
            return False
        if self.dtype != spec.dtype or ((0 < self.n < self.max_size) and (self.n + spec.n > self.max_size)):
            return False
        else:
            return True
        
    def addSpec(self, spec):
        if self.buffer is not None:
            raise ValueError("Can't add a spec to a container that already has data.")
        if self.dtype != spec.dtype:
            raise ValueError(f"Can't add ArraySpec of dtype {spec.dtype} to container of dtype {self.dtype}")
        assert self.fits(spec)
        self.specs.append(spec)
        self.n += spec.n
        self.spec_slices.append(slice(self.n - spec.n, self.n))
        return len(self.spec_slices) - 1
        
    def setData(self, array_ix, data):
        assert (data.shape[1:] == self.specs[array_ix].shape)
        assert (data.dtype == self.dtype)
        if self.buffer is None:
            self.allocate(data.shape[0])
        else:
            assert self.buffer.shape[0] == data.shape[0] # length check
        self.buffer[:, self.spec_slices[array_ix]] = data.reshape(-1, self.specs[array_ix].n)
    
    def allocate(self, length):
        self.length = length
        self.buffer = np.zeros((self.length, self.n), dtype=self.dtype)
        
    def toDisk(self, file_path, length=None, overwrite = False):
        if self.buffer is None:
            raise ValueError("Can't mmap (write) a container that has no data.")
        if isinstance(self.buffer, np.memmap):
            warnings.warn("Buffer is already memory mapped.")
        if os.path.exists(file_path) and not overwrite:
            raise ValueError("Error writing ArrayContainer to disk: file {file_path} already exists.")
        buf = self.buffer[slice(length)]
        new_buffer = np.memmap(file_path, dtype=self.dtype, shape=buf.shape, mode='w+')
        new_buffer[:] = buf[:]
        new_buffer.flush()
        del buf, new_buffer, self.buffer
        self.buffer = None
        return self.fromDisk(file_path)
        
    def fromDisk(self, file_path):
        if self.buffer is not None:
            raise ValueError("Can't mmap (read) a container that already has data.")
        self.buffer = np.memmap(file_path, dtype=self.dtype, mode='r+')
        self.buffer = self.buffer.reshape((-1, self.n))
        self.length = self.buffer.shape[0]
        return self
    
    def toMemory(self):
        if self.buffer is not None and isinstance(self.buffer, np.memmap):
            self.buffer = np.copy(self.buffer)
        return self

    def __getitem__(self, ix):
        if self.buffer is None:
            raise ValueError("No data in buffer.")
        if isinstance(ix, (int, slice)):
            ix = (ix, slice(None, None, None))
            
        data = self.buffer[ix[0]]
        if isinstance(ix[0], (slice, list)):
            if isinstance(ix[1], (slice, list)):
                return [spec.coerce(data[:,sl]) for spec, sl in zip(self.specs[ix[1]], self.spec_slices[ix[1]])]
            elif isinstance(ix[1], int):
                return self.specs[ix[1]].coerce(data[:,self.spec_slices[ix[1]]])
            raise IndexError(f"Confused by index {ix}; specifically the '{ix[1]}' bit.")
        elif isinstance(ix[0], int):
            if isinstance(ix[1], (slice, list)):
                return [spec.coerce(data[sl]) for spec, sl in zip(self.specs[ix[1]], self.spec_slices[ix[1]])]
            elif isinstance(ix[1], int):
                return self.specs[ix[1]].coerce(data[self.spec_slices[ix[1]]])
            raise IndexError(f"Confused by index {ix}; specifically the '{ix[1]}' bit.")


class ArrayCollection:
    VERSION = '0.0'
    def __init__(self, specs):
        self.containers = []
        self.map = []
        self.specs = specs
        for spec in specs:
            self.addSpec(spec)
        self.insertion_index = None
        self.length = None
        self.empty = True
    
    def __len__(self):
        return self.insertion_index or self.length
    
    def addSpec(self, spec):
        for k, container in enumerate(self.containers):
            if container.fits(spec):
                break
        else:
            container = ArrayContainer(spec.dtype)
            k = len(self.containers)
            self.containers.append(container)
        self.map.append((k, container.addSpec(spec)))
    
    def toDisk(self, root, overwrite=False):
        os.makedirs(root)
        for k, c in enumerate(self.containers):
            c.toDisk(os.path.join(root, f'{k}.{self.VERSION}.dat'), length=self.insertion_index, overwrite=overwrite)
    
    def fromDisk(self, root):
        for k, c in enumerate(self.containers):
            c.fromDisk(os.path.join(root, f'{k}.{self.VERSION}.dat'))
        self.empty = False
            
    def toMemory(self, root):
        for k, c in enumerate(self.containers):
            c.toMemory()

    def setData(self, array_ix, data):
        assert isinstance(array_ix, int)
        c, k = self.map[array_ix]
        self.containers[c].setData(k, data)
        self.empty = False
    
    def allocate(self, length):
        self.insertion_index = 0
        for c in self.containers:
            c.allocate(length)
        self.length = length
        self.empty = False
        
    def append(self, values):
        if self.insertion_index is None:
            raise ValueError("Can't insert values into an ArrayCollection without allocating first.")
        if len(values) != len(self.map):
            raise ValueError("Can only insert values N at a time (where N is the number of arrays in the collection).")
        if self.insertion_index >= self.length:
            warnings.warn("Collection is full. Skipping insertion.")
        else:
            for val, (container, ix_in_container) in zip(values, self.map):
                self.containers[container][self.insertion_index, ix_in_container][()] = val
            self.insertion_index += 1

    def __getitem__(self, ix):
        if isinstance(ix, (int, slice, list)):
            ix = (ix, slice(None,None,None))
        
        if isinstance(ix[1], (slice, list)): # using multiple containers
            m = self.map[ix[1]] # (list of (container_no, ix_in_container))
            data_ = {i:self.containers[i][ix[0]] for i in set(x[0] for x in m)}
            return [data_[c][k] for c,k in m]
        elif isinstance(ix[1], int):
            c, k = self.map[ix[1]]
            return self.containers[c][ix[0], k]
        else:
            raise IndexError("??")

            

    
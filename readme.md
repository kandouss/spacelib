# spacelib

spacelib is a small collection of tools for off-policy reinforcement learning, particularly with recurrent agents and particularly for learning directly from pixel observations.

The package makes pretty heavy use of pytorch tensors and the openai gym APIs.

Features:
 - efficiently store completed episodes on disk in memory-mapped numpy arrays
 - transparently handle hierarchically structured action/observation spaces
 - simple interface for batch sampling of sequences from experience history
 - store hidden states for recurrent models

For a usage example, see the [example notebook](https://github.com/kandouss/spacelib/blob/master/examples/Batch%20sequence%20sampling.ipynb).


### TODO
- [ ] stop one-hot encoding discrete spaces
- [ ] test/add an example for hidden state storage


This project is licensed under the terms of the MIT license, but also please let me know if you find it useful!
:space:
# BaxCat

My personal implementation of [CrossCat](http://probcomp.csail.mit.edu/crosscat/) designed for prototyping and testing. The primary concern of this code is modularity---doing a math change should affect only one class or function. *It is written entirely in python, so it will be slow.*

## Running
An example can be found in `baxcat/example.py` which generates a column of clustered data for each data type currently supported, then runs sampler transitions and plots the results in real time.

## Required Modules
- numpy
- scipy
- matplotlib (for pylab)
- sklearn (for ARI)

## Functionality Notes
- Uncollapsed datatypes (denoted with the suffix `_uc`) are slower to converge.
- Some data types do not yet have `predictive_draw()` methods so they will not work with `simple_predictive_sample()`. 
- No constrained `predictive_probability()` or `predictive_draw()` (coming soon)
- No impute (coming soon)

## Known Issues
- No documentation
- Multiple views of one cluster can exist (should be equivalent except under uncollapsed types).
- Uncollapsed samplers resample component parameters during column assignment transition.

## License
The MIT License (MIT)

Copyright (c) 2014 Baxter S. Eaves Jr.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# ouijaflow

`ouijaflow` implements the probabilistic single-cell pseudotime model Ouija in Edward and Tensorflow, allowing scalable inference on large single-cell datasets. Inference is performed using reparametrization gradient variational inference.


# Getting started

At present `ouijaflow` may be in stalled via

```
pip install git+https://github.com/kieranrcampbell/ouijaflow.git
```

Fitting pseudotimes with Ouijaflow is straightforward, following the `sklearn` syntax. If `Y` is a cell-by-gene numpy array of non-negative log expression values, then the pseudotimes may be fit via

```python
from ouijaflow import ouija
oui = ouija()
oui.fit(Y)
```

The pseudotimes can be extracted using the `trajectory` function:

```python
z = oui.trajectory()
```

The gene-specific behaviour may be extracted using the `gene_behaviour` function:

```python
oui.gene_behaviour()
```

which returns a pandas data frame with interpretable gene parameters as explained in `oui.gene_behaviour.__doc__`.

For more fine-grained control over the posterior distributions, the approximating distributions may be retrieved in a dictionary using

```python
oui.approx_dists()
```


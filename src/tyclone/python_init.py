import json,sys,numpy as np

# Purpose:
# This helper is embedded and executed by tyclone --python-compatible to create
# PyClone-VI-compatible VI initialization values from a single seeded NumPy RNG.
# It is a stdin->stdout JSON filter: read shape/seed config, emit flattened
# {pi, theta, z} arrays for each restart in deterministic draw order.

# stdin JSON: {seed, num_restarts, num_clusters, num_mutations, num_samples, num_grid_points}
c=json.load(sys.stdin)
rng=np.random.default_rng(c["seed"])
K,R,N,D,G=c["num_clusters"],c["num_restarts"],c["num_mutations"],c["num_samples"],c["num_grid_points"]
ones=np.ones(K)
out=[]
for _ in range(R):
    # Keep draw order aligned with PyClone-VI: pi -> theta -> z per restart.
    pi=rng.dirichlet(ones).tolist()
    t=rng.gamma(1,1,(K,D,G));t/=t.sum(axis=2,keepdims=True)
    out.append({"pi":pi,"theta":t.ravel().tolist(),"z":rng.dirichlet(ones,N).ravel().tolist()})
# stdout JSON: {"restarts": [{"pi": [...], "theta": [...], "z": [...]}]}
json.dump({"restarts":out},sys.stdout,separators=(",",":"))

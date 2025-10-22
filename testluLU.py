from tqdm import tqdm
from luLU import CachedAbSolver, luLU
from cupyx.scipy.sparse import csr_matrix
import cupy as cp

A = csr_matrix(cp.eye(10000, dtype=cp.float64))
b = cp.random.rand(10000, dtype=cp.float64)
# solver = CachedAbSolver(A,b)
# for _ in tqdm(range(10000)):
#     b = cp.random.rand(10000, dtype=cp.float64)
#     sol = solver.solve(b)
#     cp.testing.assert_allclose(sol, b)

lol = luLU(A)
for _ in tqdm(range(10000)):
    b = cp.random.rand(10000, dtype=cp.float64)
    sol = lol.solve(b)
    cp.testing.assert_allclose(sol, b)
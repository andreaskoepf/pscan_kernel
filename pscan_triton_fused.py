import torch
import triton
import triton.language as tl


def cdiv(x, div):
    return (x + div - 1) // div


def expand_(A, X):
    if A.size(1) == 1:
        return
    # print("expand_ ", A.shape, X.shape)
    T = 2 * (A.size(1) // 2)
    Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
    Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
    # print("Aa, Xa", Aa.shape, Xa.shape)
    Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
    Aa[:, :, 1].mul_(Aa[:, :, 0])
    expand_(Aa[:, :, 1], Xa[:, :, 1])

    Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
    Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
    if T < A.size(1):
        # print('fixup:', T, A.size(1))
        X[:, -1].add_(A[:, -1].mul(X[:, -2]))
        A[:, -1].mul_(A[:, -2])



def addmulop(A, X, b1, e1, b2, e2, d, stride):
    X[:, d::stride] += A[:, b2:e2:stride] * X[:, b1:e1:stride]
    A[:, d::stride] *= A[:, b1:e1:stride]


def expand2_recursive(A, X, t, offset, stride):
    if stride > t:
        return
    addmulop(A, X, b1=offset, e1=t, b2=offset+stride//2, e2=t, d=offset+stride//2, stride=stride)
    expand2_recursive(A, X, t, offset+stride//2, stride*2)
    addmulop(A, X, b1=offset+stride//2, e1=t-stride, b2=offset+stride, e2=t, d=offset+stride, stride=stride)


def expand2_(A, X):
    expand2_recursive(A, X, A.size(1), 0, 2)


@triton.jit
def addmul_kernel(
    A,  # [N, T, 1]
    X,  # [N, T, D]
    stride_AN: int,
    stride_AT: int,
    stride_XN: int,
    stride_XT: int,
    stride_XD: int,
    b1: int, 
    e1: int, 
    b2: int,
    d: int,
    stride: int,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    T_block_size: tl.constexpr,
    D_block_size: tl.constexpr,
):
    n = tl.program_id(axis=0)
    t_chunk = tl.program_id(axis=1)
    d_chunk = tl.program_id(axis=2)

    A_base = A + n * stride_AN
    X_base = X + n * stride_XN 

    tb = tl.arange(0, T_block_size) + t_chunk * T_block_size
    seq_dest_offs = tb * stride + d
    seq_a_offs = tb * stride + b2
    seq_x_offs = tb * stride + b1
    
    dim_offs = tl.arange(0, D_block_size) + d_chunk * D_block_size

    #X[:, d::stride] += A[:, b2:e2:stride] * X[:, b1:e1:stride]
    a_mask = seq_a_offs < seqlen
    a = tl.load(A_base + seq_a_offs * stride_AT, mask=a_mask)

    x_mask = (seq_x_offs[:, None] < e1) & (dim_offs[None, :] < dim)

    x_src = tl.load(X_base + seq_x_offs[:, None] * stride_XT  + dim_offs[None, :] * stride_XD, mask=x_mask)
    x_dst = tl.load(X_base + seq_dest_offs[:, None] * stride_XT  + dim_offs[None, :] * stride_XD, mask=x_mask)

    y = x_dst + x_src * a[:, None]
    tl.store(X_base + seq_dest_offs[:, None] * stride_XT  + dim_offs[None, :] * stride_XD, y, mask=x_mask)


def addmul_triton(
    A: torch.Tensor,  # [N, T, 1]
    X: torch.Tensor,  # [N, T, D]
    b1: int, 
    e1: int, 
    b2: int, 
    d: int,
    stride: int
):
    # shape checks
    N, T, D = X.shape

    assert A.shape[0] == N, "N mismatch"
    assert A.shape[1] == T, "T mismatch"

    # compute blocks for t dimenion we compute
    num_t = cdiv(T - d, stride)

    if num_t >= 64:
        block_size_seq = 64
    elif num_t >= 32:
        block_size_seq = 32
    elif num_t >= 16:
        block_size_seq = 16
    else:
        block_size_seq = 8
    
    if D >= 64:
        block_size_dim = 64
    elif D >= 32:
        block_size_dim = 32
    elif D >= 16:
        block_size_dim = 16
    else:
        block_size_dim = 8

    num_blocks_T = cdiv(num_t, block_size_seq)
    num_blocks_D = cdiv(D, block_size_dim)

    grid = (N, num_blocks_T, num_blocks_D)
    addmul_kernel[grid](
        A,  # [N, T, 1]
        X,  # [N, T, D]
        stride_AN=A.stride(0),
        stride_AT=A.stride(1),
        stride_XN=X.stride(0),
        stride_XT=X.stride(1),
        stride_XD=X.stride(2),
        seqlen=T,
        dim=D,
        b1=b1, 
        e1=e1, 
        b2=b2,
        d=d,
        stride=stride,
        T_block_size=block_size_seq,
        D_block_size=block_size_dim,
    )


def addmulop2(A, X, b1, e1, b2, d, stride):
    addmul_triton(A, X, b1, e1, b2, d, stride)
    A[:, d::stride].mul_(A[:, b1:e1:stride])


def expand_fused_recurse(A, X, t, offset, stride):
    if stride > t:
        return
    addmulop2(A, X, b1=offset, e1=t, b2=offset+stride//2, d=offset+stride//2, stride=stride)
    expand_fused_recurse(A, X, t, offset+stride//2, stride*2)
    addmulop2(A, X, b1=offset+stride//2, e1=t-stride, b2=offset+stride, d=offset+stride, stride=stride)


def expand_fused_(A, X):
    expand_fused_recurse(A, X, A.size(1), 0, 2)


def benchmark_params(N, T, D):
    print()
    print(f"N={N}, T={T}, D={D}")

    A = torch.rand(N, T, 1, dtype=torch.float32, device="cuda") * 0.01 + 1
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")

    # generate reference results
    A0, X0 = A.clone(), X.clone()
    expand_(A0, X0)

    # check graph version
    
    # A5,X5 = expand_replay(A, X)
    # print("torch.allclose(A0, A5)", torch.allclose(A0, A5), torch.max(torch.abs(A0-A5)).item())
    # print("torch.allclose(X0, X5)", torch.allclose(X0, X5), torch.max(torch.abs(X0-X5)).item())

    @torch.inference_mode()
    @torch.no_grad()
    def benchmark_single(name, expand_fn, runs: int=3, warmup: int=1):

        for i in range(warmup):
            for _ in range(100):
                A1 = A.clone()
                X1 = X.clone()
                expand_fn(A1, X1)

        times = []
        for i in range(runs):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                A1 = A.clone()
                X1 = X.clone()
                expand_fn(A1, X1)
            end.record()
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            times.append(elapsed)

        print(f"{name}: {sum(times)/runs:.2f}ms")
        return A1, X1

    A1, X1 = benchmark_single("torch eager", expand_)
    A2, X2 = benchmark_single("triton fused", expand_fused_)

    if not torch.allclose(A1, A2) or not torch.allclose(X1, X2):
        print("ERROR: Results don't match!")
        print(
            "torch.allclose(A1, A2)",
            torch.allclose(A1, A2),
            torch.max(torch.abs(A1 - A2)).item(),
        )
        print(
            "torch.allclose(X1, X2)",
            torch.allclose(X1, X2),
            torch.max(torch.abs(X1 - X2)).item(),
        )


def benchmark():
    configs = (
        (2, 1024, 8),
        (26, 16, 32),
        (128, 1024, 32),
        (384, 1024, 256),
        (512, 2048, 256),
    )
    
    for N, T, D in configs:
        benchmark_params(N, T, D)


def dev():
    torch.manual_seed(42)

    N, T, D = 2, 32, 64
    A = torch.randn(N, T, 1, dtype=torch.float32, device="cuda") * 0.1 + 1
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")

    A1 = A.clone()
    X1 = X.clone()

    expand_(A1, X1)

    A2 = A.clone()
    X2 = X.clone()
    #expand2_(A2, X2)
    expand_fused_(A2, X2)
    
    torch.cuda.synchronize()

    # print("A2-A1", A2 - A1)
    dx = X2 - X1
    dx[dx.abs() < 0.0001] = 0

    torch.set_printoptions(profile="full")
    #print("A2-A1", A2-A1)
    print("X2-X1", dx[0, 0:, 0:])

    #print("X2", X2[0, 0:, 0:])
    print()

    print("A == A_", torch.allclose(A1, A2), torch.max(torch.abs(A1 - A2)).item())
    print("X == X_", torch.allclose(X1, X2), torch.max(torch.abs(X1 - X2)).item())


if __name__ == '__main__':
    benchmark()

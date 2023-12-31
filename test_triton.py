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


@triton.jit
def expand_kernel(
    A,  # [N, T, 1]
    X,  # [N, T, D]
    stride_AN: int,
    stride_AT: int,
    stride_AD: int,
    stride_XN: int,
    stride_XT: int,
    stride_XD: int,
    batch: tl.constexpr,
    seqlen: tl.constexpr,
    dim: tl.constexpr,
    T_block_size: tl.constexpr,
    D_block_size: tl.constexpr,
):
    n = tl.program_id(axis=0)
    dim_chunk = tl.program_id(axis=1)

    A_base = A + n * stride_AN + stride_AD * dim_chunk
    X_base = X + n * stride_XN 

    offs_dim = tl.arange(0, D_block_size) + dim_chunk * D_block_size

    view_stride = 1
    view_offset = 0
    while view_offset + view_stride < seqlen:
        indices0 = tl.arange(0, T_block_size) * 2 * view_stride
        indices1 = indices0 + view_stride

        block_offset = view_offset
        while block_offset < seqlen:
            # read values
            
            a1 = tl.load(A_base + (indices1 + block_offset) * stride_AT)

            # load block T_block_size x D_block_size
            # Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            x0 = tl.load(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD
            )
            x1 = tl.load(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD
            )
            x1 += (x0 * a1[:, None])
            x_mask = ((indices1 + block_offset)[:, None] < seqlen) & (offs_dim[None, :] < dim)
            tl.store(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                x1,
                mask=x_mask,
            )

            # Aa[:, :, 1].mul_(Aa[:, :, 0])
            a0 = tl.load(A_base + (indices0 + block_offset) * stride_AT)
            b = a0 * a1

            # store
            a_mask = (indices1 + block_offset) < seqlen
            tl.store(A_base + (indices1 + block_offset) * stride_AT, b, mask=a_mask)

            block_offset += T_block_size * view_stride

        view_offset += view_stride
        view_stride = view_stride * 2

    view_stride = view_stride // 2
    view_offset -= view_stride

    # downward pass
    while view_stride > 0:
        indices1 = tl.arange(0, T_block_size) * 2 * view_stride + view_stride
        indices0 = indices1 + view_stride

        block_offset = view_offset
        while block_offset < seqlen:
            # read values
            a0 = tl.load(A_base + (indices0 + block_offset) * stride_AT)
            
            # Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
            x0 = tl.load(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD
            )
            x1 = tl.load(
                X_base
                + (indices1 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD
            )
            x0 += (x1 * a0[:, None])
            x_mask = ((indices0 + block_offset)[:, None] < seqlen) & (offs_dim[None, :] < dim)
            tl.store(
                X_base
                + (indices0 + block_offset)[:, None] * stride_XT
                + offs_dim[None, :] * stride_XD,
                x0,
                mask=x_mask,
            )

            # Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
            a1 = tl.load(A_base + (indices1 + block_offset) * stride_AT)
            b = a0 * a1

            # store
            mask = (indices0 + block_offset) < seqlen
            tl.store(
                A_base + (indices0 + block_offset) * stride_AT, b, mask=mask
            )

            block_offset += T_block_size * view_stride

        view_stride = view_stride // 2
        view_offset -= view_stride


def expand_triton(
    A: torch.Tensor,  # [N, T, 1]
    X: torch.Tensor,  # [N, T, D]
):
    # shape checks
    N, T, D = X.shape

    assert A.shape[0] == N, "N mismatch"
    assert A.shape[1] == T, "T mismatch"

    block_size_dim = 32
    block_size_seq = 32

    dim_blocks = cdiv(D, block_size_dim)

    # tepmorary expansion of A for temp storage
    A_ = A.repeat(1, 1, cdiv(D, block_size_dim)).contiguous()

    grid = (N, dim_blocks)
    expand_kernel[grid](
        A_,  # [N, T, dim_blocks]
        X,  # [N, T, D]
        stride_AN=A_.stride(0),
        stride_AT=A_.stride(1),
        stride_AD=A_.stride(2),
        stride_XN=X.stride(0),
        stride_XT=X.stride(1),
        stride_XD=X.stride(2),
        batch=N,
        seqlen=T,
        dim=D,
        T_block_size=block_size_seq,
        D_block_size=block_size_dim,
    )
    A.copy_(A_[:, :, :1])



def benchmark():
    #N, T, D = 384, 1024, 64
    N, T, D = 384, 128, 64

    A = torch.rand(N, T, 1, dtype=torch.float32, device="cuda") + 0.55
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")

    # generate reference results
    A0, X0 = A.clone(), X.clone()
    expand_(A0, X0)

    @torch.inference_mode()
    @torch.no_grad()
    def benchmark(name, expand_fn):
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
        print(f"{name}: {elapsed:.4}ms")
        return A1, X1

    for i in range(3):
        A1, X1 = benchmark("torch eager", expand_)
    for i in range(3):
        A2, X2 = benchmark("custom cuda", expand_triton)

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

def main():
    torch.manual_seed(42)

    N, T, D = 1, 32, 64
    A = torch.randn(N, T, 1, dtype=torch.float32, device="cuda") * 0.1 + 1
    X = torch.rand(N, T, D, dtype=torch.float32, device="cuda")

    A1 = A.clone()
    X1 = X.clone()

    expand_(A1, X1)

    A2 = A.clone()
    X2 = X.clone()
    expand_triton(A2, X2)
    torch.cuda.synchronize()

    # print("A2-A1", A2 - A1)
    dx = X2 - X1
    dx[dx.abs() < 0.0001] = 0

    torch.set_printoptions(profile="full")
    print("A2-A1", A2-A1)
    print("X2-X1", dx[0, 0:, 0:])

    #print("X2", X2[0, 0:, 0:])
    print()

    print("A == A_", torch.allclose(A1, A2), torch.max(torch.abs(A1 - A2)).item())
    print("X == X_", torch.allclose(X1, X2), torch.max(torch.abs(X1 - X2)).item())


if __name__ == "__main__":
    main()

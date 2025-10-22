import cupy as _cupy
import numpy as _numpy
import cupyx.scipy.sparse
from cupyx.scipy.sparse.linalg import SuperLU
from cupy_backends.cuda.api import driver as _driver
from cupy_backends.cuda.api import runtime as _runtime
from cupy_backends.cuda.libs import cusparse as _cusparse
from cupy._core import _dtype
from cupy.cuda import device as _device
from cupy.cuda import stream as _stream
from cupyx.cusparse import _dtype_to_IndexType, SpMatDescriptor, DnMatDescriptor, check_availability

class CachedAbSolver:
    
    def __init__(self, a, b, alpha=1.0, lower=True, unit_diag=False, transa=False):
        if not check_availability('spsm'):
            raise RuntimeError('spsm is not available.')

        # Canonicalise transa
        if transa is False:
            transa = 'N'
        elif transa is True:
            transa = 'T'
        elif transa not in 'NTH':
            raise ValueError(f'Unknown transa (actual: {transa})')

        # Check A's type and sparse format
        if cupyx.scipy.sparse.isspmatrix_csr(a):
            pass
        elif cupyx.scipy.sparse.isspmatrix_csc(a):
            if transa == 'N':
                a = a.T
                transa = 'T'
            elif transa == 'T':
                a = a.T
                transa = 'N'
            elif transa == 'H':
                a = a.conj().T
                transa = 'N'
            lower = not lower
        elif cupyx.scipy.sparse.isspmatrix_coo(a):
            pass
        else:
            raise ValueError('a must be CSR, CSC or COO sparse matrix')
        assert a.has_canonical_format

        # Check B's ndim
        if b.ndim == 1:
            is_b_vector = True
            b = b.reshape(-1, 1)
        elif b.ndim == 2:
            is_b_vector = False
        else:
            raise ValueError('b.ndim must be 1 or 2')
        self.is_b_vector = is_b_vector

        # Check shapes
        if not (a.shape[0] == a.shape[1] == b.shape[0]):
            raise ValueError('mismatched shape')

        # Check dtypes
        dtype = a.dtype
        if dtype.char not in 'fdFD':
            raise TypeError('Invalid dtype (actual: {})'.format(dtype))
        if dtype != b.dtype:
            raise TypeError('dtype mismatch')
        self.dtype = dtype

        # Prepare fill mode
        if lower is True:
            fill_mode = _cusparse.CUSPARSE_FILL_MODE_LOWER
        elif lower is False:
            fill_mode = _cusparse.CUSPARSE_FILL_MODE_UPPER
        else:
            raise ValueError('Unknown lower (actual: {})'.format(lower))
        self.fill_mode = fill_mode

        # Prepare diag type
        if unit_diag is False:
            diag_type = _cusparse.CUSPARSE_DIAG_TYPE_NON_UNIT
        elif unit_diag is True:
            diag_type = _cusparse.CUSPARSE_DIAG_TYPE_UNIT
        else:
            raise ValueError('Unknown unit_diag (actual: {})'.format(unit_diag))
        self.diag_type = diag_type

        # Prepare op_a
        if transa == 'N':
            op_a = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        elif transa == 'T':
            op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:  # transa == 'H'
            if dtype.char in 'fd':
                op_a = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
            else:
                op_a = _cusparse.CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE
        self.op_a = op_a
        # Prepare op_b
        self.op_b = self._get_opb(b)
        
        # Allocate space for matrix C. Note that it is known cusparseSpSM requires
        # the output matrix zero initialized.
        m, _ = a.shape
        if self.op_b == _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE:
            _, n = b.shape
        else:
            n, _ = b.shape
        c_shape = m, n
        self.c_shape = c_shape

        self._perform_analysis(a, b, alpha=alpha)

    def _perform_analysis(self, a, b, alpha=1.0):
        """Solves a sparse triangular linear system op(a) * x = alpha * op(b).

        Args:
            a (cupyx.scipy.sparse.csr_matrix or cupyx.scipy.sparse.coo_matrix):
                Sparse matrix with dimension ``(M, M)``.
            b (cupy.ndarray): Dense matrix with dimension ``(M, K)``.
            alpha (float or complex): Coefficient.
            lower (bool):
                True: ``a`` is lower triangle matrix.
                False: ``a`` is upper triangle matrix.
            unit_diag (bool):
                True: diagonal part of ``a`` has unit elements.
                False: diagonal part of ``a`` has non-unit elements.
            transa (bool or str): True, False, 'N', 'T' or 'H'.
                'N' or False: op(a) == ``a``.
                'T' or True: op(a) == ``a.T``.
                'H': op(a) == ``a.conj().T``.
        """
        
        c = _cupy.zeros(self.c_shape, dtype=a.dtype, order='f')

        # Prepare descriptors and other parameters
        self.handle = _device.get_cusparse_handle()
        self.mat_a = SpMatDescriptor.create(a)
        mat_b = DnMatDescriptor.create(b)
        mat_c = DnMatDescriptor.create(c)
        self.spsm_descr = _cusparse.spSM_createDescr()
        self.alpha = _numpy.array(alpha, dtype=c.dtype).ctypes
        self.cuda_dtype = _dtype.to_cuda_dtype(c.dtype)
        self.algo = _cusparse.CUSPARSE_SPSM_ALG_DEFAULT

        try:
            # Specify Lower|Upper fill mode
            self.mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_FILL_MODE, self.fill_mode)

            # Specify Unit|Non-Unit diagonal type
            self.mat_a.set_attribute(_cusparse.CUSPARSE_SPMAT_DIAG_TYPE, self.diag_type)

            # Allocate the workspace needed by the succeeding phases
            buff_size = _cusparse.spSM_bufferSize(
                self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
                mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr)
            self.buff = _cupy.empty(buff_size, dtype=_cupy.int8)

            # Perform the analysis phase
            _cusparse.spSM_analysis(
                self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
                mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr, self.buff.data.ptr)
        except:
            raise RuntimeError('spSM_analysis failed.')
    
    def _get_opb(self, b):
        # Prepare op_b
        if b._f_contiguous:
            op_b = _cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        elif b._c_contiguous:
            if _cusparse.get_build_version() < 11701:  # earlier than CUDA 11.6
                raise ValueError('b must be F-contiguous.')
            b = b.T
            op_b = _cusparse.CUSPARSE_OPERATION_TRANSPOSE
        else:
            raise ValueError('b must be F-contiguous or C-contiguous.')
        return op_b
    
    def solve(self, b):
        if b.ndim == 1:
            is_b_vector = True
            b = b.reshape(-1, 1)
        elif b.ndim == 2:
            is_b_vector = False
        else:
            raise ValueError('b.ndim must be 1 or 2')
        self.is_b_vector = is_b_vector
        c = _cupy.zeros(self.c_shape, dtype=self.dtype, order='f')
        mat_b = DnMatDescriptor.create(b)
        mat_c = DnMatDescriptor.create(c)
        # Executes the solve phase
        _cusparse.spSM_solve(
            self.handle, self.op_a, self.op_b, self.alpha.data, self.mat_a.desc, mat_b.desc,
            mat_c.desc, self.cuda_dtype, self.algo, self.spsm_descr, self.buff.data.ptr)

        # Reshape back if B was a vector
        if self.is_b_vector:
            c = c.reshape(-1)
        return c

    # def __del__(self):
    #     # Destroy matrix descriptor
    #     print("Deleting solver obj...")
    #     _cusparse.spSM_destroyDescr(self.spsm_descr)

class luLU(SuperLU):

    def __init__(self, obj, b_dtype = _cupy.float64):
        super().__init__(obj)
        self.b_dtype = b_dtype
        b_sample = _cupy.ones(self.shape[0]*self.shape[1],dtype=self.b_dtype)

    
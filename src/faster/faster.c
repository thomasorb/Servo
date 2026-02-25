#define NPY_NO_DEPRECATED_API NPY_1_19_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

/* ============================================================
   Helper macros
   ============================================================ */

static int ensure_float32_2d(PyArrayObject* a) {
    /* Ensure: 2D, float32, C‑contiguous */
    return PyArray_NDIM(a)==2 &&
           PyArray_TYPE(a)==NPY_FLOAT32 &&
           PyArray_ISCONTIGUOUS(a);
}

static int ensure_float32_1d(PyArrayObject* a) {
    /* Ensure: 1D, float32, C‑contiguous */
    return PyArray_NDIM(a)==1 &&
           PyArray_TYPE(a)==NPY_FLOAT32 &&
           PyArray_ISCONTIGUOUS(a);
}

static int ensure_int32_1d(PyArrayObject* a) {
    /* Ensure: 1D, int32, C‑contiguous */
    return PyArray_NDIM(a)==1 &&
           PyArray_TYPE(a)==NPY_INT32 &&
           PyArray_ISCONTIGUOUS(a);
}


/* ============================================================
   1) compute_profiles_local_f32
      ----------------------------------------------------------
      Computes horizontal and vertical profiles around (x,y)
      using a local (non‑integral) mean over a strip.

      Horizontal:
        hprofile[i] = mean of row r = x - hl + i
                       over columns [y-hw, y+hw)

      Vertical:
        vprofile[i] = mean of column c = y - hl + i
                       over rows [x-hw, x+hw)

      ROI:
        Returns a rectangular region [x-hl:x+hl, y-hl:y+hl]
        truncated to image boundaries.

      Requirements:
        - a: float32 2D array
        - w and l must be even
        - returns (vprofile, hprofile, roi)
   ============================================================ */

static PyObject* py_compute_profiles_local_f32(
    PyObject* self, PyObject* args, PyObject* kwargs)
{
    PyObject *a_obj = NULL;
    int x, y, w, l;
    int get_roi = 1;

    static char* kwlist[] = {"a","x","y","w","l","get_roi",NULL};

    if (!PyArg_ParseTupleAndKeywords(
            args, kwargs, "Oiiii|p",
            kwlist, &a_obj, &x, &y, &w, &l, &get_roi))
        return NULL;

    /* Convert to float32 contiguous if needed */
    PyArrayObject* a = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED);
    if (!a) return NULL;

    if (PyArray_NDIM(a) != 2) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "a must be 2D float32 array");
        return NULL;
    }

    npy_intp nrows = PyArray_DIM(a,0);
    npy_intp ncols = PyArray_DIM(a,1);
    float* A = (float*)PyArray_DATA(a);

    if ((w & 1) || (l & 1)) {
        Py_DECREF(a);
        PyErr_SetString(PyExc_ValueError, "w and l must be even");
        return NULL;
    }

    int hw = w/2;
    int hl = l/2;

    /* Allocate output profiles */
    npy_intp dims[1] = {l};
    PyArrayObject* hprof =
        (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    PyArrayObject* vprof =
        (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);

    if (!hprof || !vprof) {
        Py_XDECREF(hprof);
        Py_XDECREF(vprof);
        Py_DECREF(a);
        return NULL;
    }

    float* HP = (float*)PyArray_DATA(hprof);
    float* VP = (float*)PyArray_DATA(vprof);

    /* Compute ROI bounds */
    int r0 = x - hl;
    int r1 = x + hl;
    int c0 = y - hl;
    int c1 = y + hl;

    int rr0 = (r0 < 0) ? 0 : r0;
    int rr1 = (r1 > (int)nrows) ? (int)nrows : r1;
    int cc0 = (c0 < 0) ? 0 : c0;
    int cc1 = (c1 > (int)ncols) ? (int)ncols : c1;

    /* ---- Compute profiles ---- */
    Py_BEGIN_ALLOW_THREADS

    for (int i = 0; i < l; ++i) {

        /* -------- Horizontal profile -------- */
        int r = x - hl + i;
        if (r >= 0 && r < (int)nrows) {
            int c0i = (y - hw < 0) ? 0 : (y - hw);
            int c1i = (y + hw > (int)ncols) ? (int)ncols : (y + hw);
            int width = c1i - c0i;

            if (width <= 0) {
                HP[i] = NPY_NAN;
            } else {
                const float* row = A + ((npy_intp)r) * ncols;
                float sum = 0.0f;
                for (int c = c0i; c < c1i; ++c) sum += row[c];
                HP[i] = sum / (float) width;
            }
        } else {
            HP[i] = NPY_NAN;
        }

        /* -------- Vertical profile -------- */
        int c = y - hl + i;
        if (c >= 0 && c < (int)ncols) {
            int r0i = (x - hw < 0) ? 0 : (x - hw);
            int r1i = (x + hw > (int)nrows) ? (int)nrows : (x + hw);
            int height = r1i - r0i;

            if (height <= 0) {
                VP[i] = NPY_NAN;
            } else {
                float sum = 0.0f;
                for (int r2 = r0i; r2 < r1i; ++r2)
                    sum += A[((npy_intp)r2) * ncols + c];
                VP[i] = sum / (float) height;
            }
        } else {
            VP[i] = NPY_NAN;
        }
    }

    Py_END_ALLOW_THREADS

    /* ---- Build ROI (if requested) ---- */
    PyArrayObject* roi = NULL;

    if (get_roi) {
        npy_intp rdims[2] = { rr1 - rr0, cc1 - cc0 };
        roi = (PyArrayObject*)PyArray_SimpleNew(2, rdims, NPY_FLOAT32);
        if (!roi) {
            Py_DECREF(a);
            Py_DECREF(hprof);
            Py_DECREF(vprof);
            return NULL;
        }

        float* R = (float*)PyArray_DATA(roi);

        Py_BEGIN_ALLOW_THREADS
        for (int r = 0; r < (rr1 - rr0); ++r) {
            const float* src =
                A + ((npy_intp)(rr0 + r)) * ncols + cc0;
            float* dst = R + ((npy_intp)r) * (cc1 - cc0);
            for (int c = 0; c < (cc1 - cc0); ++c)
                dst[c] = src[c];
        }
        Py_END_ALLOW_THREADS
    }

    Py_DECREF(a);

    if (get_roi)
        return Py_BuildValue("NNN", vprof, hprof, roi);
    else
        return Py_BuildValue("NN", vprof, hprof);
}


/* ============================================================
   2) batch_normalize_and_levels
      ----------------------------------------------------------
      Computes 3 normalized/clamped mean levels over 3 pixel
      index lists (left, center, right).

      Equivalent to calling normalize_and_compute_profile_level
      three times, but done in a compact C loop.

      Requirements:
        - a, vmin, vmax: float32 1D
        - left, center, right: int32 1D
        - returns float32[3]
   ============================================================ */

static PyObject* py_batch_normalize_and_levels(
    PyObject* self, PyObject* args)
{
    PyObject *a_obj, *vmin_obj, *vmax_obj;
    PyObject *left_obj, *center_obj, *right_obj;

    if (!PyArg_ParseTuple(
            args, "OOOOOO",
            &a_obj, &vmin_obj, &vmax_obj,
            &left_obj, &center_obj, &right_obj))
        return NULL;

    PyArrayObject *a =
        (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *vmin =
        (PyArrayObject*)PyArray_FROM_OTF(vmin_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *vmax =
        (PyArrayObject*)PyArray_FROM_OTF(vmax_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);

    PyArrayObject *left =
        (PyArrayObject*)PyArray_FROM_OTF(left_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *center =
        (PyArrayObject*)PyArray_FROM_OTF(center_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject *right =
        (PyArrayObject*)PyArray_FROM_OTF(right_obj, NPY_INT32, NPY_ARRAY_C_CONTIGUOUS);

    if (!a || !vmin || !vmax || !left || !center || !right) {
        Py_XDECREF(a); Py_XDECREF(vmin); Py_XDECREF(vmax);
        Py_XDECREF(left); Py_XDECREF(center); Py_XDECREF(right);
        return NULL;
    }

    if (!ensure_float32_1d(a) ||
        !ensure_float32_1d(vmin) ||
        !ensure_float32_1d(vmax) ||
        !ensure_int32_1d(left) ||
        !ensure_int32_1d(center) ||
        !ensure_int32_1d(right))
    {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "Invalid array shapes or dtypes");
        return NULL;
    }

    if (PyArray_DIM(a,0) != PyArray_DIM(vmin,0) ||
        PyArray_DIM(a,0) != PyArray_DIM(vmax,0))
    {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        PyErr_SetString(PyExc_ValueError, "Size mismatch in a/vmin/vmax");
        return NULL;
    }

    float *A    = (float*)PyArray_DATA(a);
    float *VMIN = (float*)PyArray_DATA(vmin);
    float *VMAX = (float*)PyArray_DATA(vmax);

    int *L = (int*)PyArray_DATA(left);
    int *C = (int*)PyArray_DATA(center);
    int *R = (int*)PyArray_DATA(right);

    npy_intp nL = PyArray_DIM(left,0);
    npy_intp nC = PyArray_DIM(center,0);
    npy_intp nR = PyArray_DIM(right,0);

    npy_intp odims[1] = {3};
    PyArrayObject* out =
        (PyArrayObject*)PyArray_SimpleNew(1, odims, NPY_FLOAT32);
    if (!out) {
        Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
        Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);
        return NULL;
    }

    float* O = (float*)PyArray_DATA(out);

    Py_BEGIN_ALLOW_THREADS

    for (int k = 0; k < 3; ++k) {

        int* IDX = (k == 0 ? L : (k == 1 ? C : R));
        npy_intp n = (k == 0 ? nL : (k == 1 ? nC : nR));

        if (n == 0) { O[k] = 0.0f; continue; }

        double acc = 0.0;

        for (npy_intp j = 0; j < n; ++j) {
            int i = IDX[j];
            float denom = VMAX[i] - VMIN[i];
            float x = (denom <= 0.0f) ? 0.0f : (A[i] - VMIN[i]) / denom;
            if (x < 0.0f) x = 0.0f;
            else if (x > 1.0f) x = 1.0f;
            acc += (double) x;
        }

        O[k] = (float)(acc / (double)n);
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(a); Py_DECREF(vmin); Py_DECREF(vmax);
    Py_DECREF(left); Py_DECREF(center); Py_DECREF(right);

    return (PyObject*) out;
}


/* ============================================================
   3) fast_copy_transpose
      ----------------------------------------------------------
      Copies src_2d.T.flatten() into a preallocated 1D float32.

      Requirements:
        - a: float32 2D
        - out: float32 1D of size nrows*ncols

        This respects your viewer orientation rules while
        avoiding Python/NumPy temporary allocations.
   ============================================================ */

static PyObject* py_fast_copy_transpose(PyObject* self, PyObject* args)
{
    PyObject *a_obj, *out_obj;

    if (!PyArg_ParseTuple(args, "OO", &a_obj, &out_obj))
        return NULL;

    PyArrayObject* a =
        (PyArrayObject*)PyArray_FROM_OTF(a_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);
    PyArrayObject* out =
        (PyArrayObject*)PyArray_FROM_OTF(out_obj, NPY_FLOAT32, NPY_ARRAY_C_CONTIGUOUS);

    if (!a || !out) {
        Py_XDECREF(a);
        Py_XDECREF(out);
        return NULL;
    }

    if (PyArray_NDIM(a) != 2 || PyArray_NDIM(out) != 1) {
        Py_DECREF(a); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "Invalid dimensions");
        return NULL;
    }

    npy_intp nrows = PyArray_DIM(a,0);
    npy_intp ncols = PyArray_DIM(a,1);

    if (PyArray_DIM(out,0) != nrows * ncols) {
        Py_DECREF(a); Py_DECREF(out);
        PyErr_SetString(PyExc_ValueError, "Output buffer has incorrect size");
        return NULL;
    }

    float* A = (float*)PyArray_DATA(a);
    float* O = (float*)PyArray_DATA(out);

    /* Column-major traversal of A, which gives A.T.ravel() */
    Py_BEGIN_ALLOW_THREADS

    npy_intp k = 0;
    for (npy_intp c = 0; c < ncols; ++c) {
        const float* col_start = A + c; /* row 0, column c */
        for (npy_intp r = 0; r < nrows; ++r) {
            /* A[r, c] = col_start[r * ncols] */
            O[k++] = col_start[r * ncols];
        }
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(a);
    Py_DECREF(out);

    Py_RETURN_NONE;
}


/* ============================================================
   Module definition
   ============================================================ */

static PyMethodDef Methods[] = {
    {"compute_profiles_local_f32",
        (PyCFunction)py_compute_profiles_local_f32,
        METH_VARARGS | METH_KEYWORDS,
        "Compute H/V profiles and ROI from a float32 2D image"},

    {"batch_normalize_and_levels",
        py_batch_normalize_and_levels, METH_VARARGS,
        "Compute 3 normalized clamped means over pixel index lists"},

    {"fast_copy_transpose",
        py_fast_copy_transpose, METH_VARARGS,
        "Copy a.T.ravel() into an existing float32 1D buffer"},

    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "faster",
    "Fast C kernels for IRCam",
    -1,
    Methods
};

PyMODINIT_FUNC PyInit_faster(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
